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
                    <a href="{{ base_path }}/terms-of-service/">Terms of Service</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <article class="blog-post">
               <header class="post-header">
                    <h1>{{ post.title }}</h1>
                    <div class="post-meta">
                        <time datetime="{{ post.created_at }}">{{ post.created_at.split('T')[0] }}</time>
                    </div>
                    {% if post.tags %}
                    <div class="tags">
                        {% for tag in post.tags[:6] %}
                        <span class="tag">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
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
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
            </div>
        </footer>
        <!-- Enhanced Navigation Script -->
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
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
                <div class="social-links">
                    {% for platform, url in social_links.items() %}
                    <a href="{{ url }}" target="_blank" rel="noopener">{{ platform|title }}</a>
                    {% endfor %}
                </div>
            </div>
        </footer>
            <!-- Enhanced Navigation Script -->
        <script src="{{ base_path }}/static/navigation.js"></script>
        
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
        <style>
            .about-section {
                background: #f8f9fa;
                padding: 2rem;
                margin-bottom: 2rem;
                border-radius: 8px;
                border-left: 4px solid #6366f1;
            }
            
            .about-section h2 {
                color: #333;
                margin-top: 0;
                margin-bottom: 1rem;
                font-size: 1.5rem;
            }
            
            .about-section h3 {
                color: #555;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
                font-size: 1.2rem;
            }
            
            .about-section p {
                line-height: 1.8;
                margin-bottom: 1rem;
            }
            
            .about-section ul {
                margin-left: 1.5rem;
                line-height: 1.8;
            }
            
            .about-section ul li {
                margin-bottom: 0.5rem;
            }
            
            .mission-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 8px;
                margin: 2rem 0;
                text-align: center;
            }
            
            .mission-box h2 {
                margin-top: 0;
                color: white;
                font-size: 2rem;
            }
            
            .mission-box p {
                font-size: 1.1rem;
                line-height: 1.8;
                margin-bottom: 0;
            }
            
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .feature-card {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                border: 2px solid #e0e0e0;
                transition: transform 0.3s ease, border-color 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                border-color: #6366f1;
            }
            
            .feature-card h3 {
                color: #6366f1;
                margin-top: 0;
                margin-bottom: 1rem;
                font-size: 1.3rem;
            }
            
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
                display: block;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .stat-card {
                background: #f0f4ff;
                padding: 1.5rem;
                border-radius: 8px;
                text-align: center;
                border: 2px solid #6366f1;
            }
            
            .stat-number {
                font-size: 2.5rem;
                font-weight: bold;
                color: #6366f1;
                display: block;
                margin-bottom: 0.5rem;
            }
            
            .stat-label {
                color: #555;
                font-size: 1rem;
            }
            
            .topics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            }
            
            .topic-tag {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #8b5cf6;
                font-weight: 500;
                color: #333;
            }
            
            .cta-box {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 2rem;
                border-radius: 8px;
                margin: 2rem 0;
                text-align: center;
            }
            
            .cta-box h3 {
                margin-top: 0;
                color: #333;
                font-size: 1.5rem;
            }
            
            .cta-button {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem 2rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                margin-top: 1rem;
                transition: transform 0.3s ease;
            }
            
            .cta-button:hover {
                transform: scale(1.05);
            }
            
            .team-section {
                background: #f0f4ff;
                padding: 2rem;
                border-radius: 8px;
                border-left: 4px solid #8b5cf6;
            }
            
            .timeline {
                position: relative;
                padding-left: 2rem;
                margin: 2rem 0;
            }
            
            .timeline-item {
                position: relative;
                padding-bottom: 2rem;
            }
            
            .timeline-item::before {
                content: '';
                position: absolute;
                left: -2rem;
                top: 0;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #6366f1;
            }
            
            .timeline-item::after {
                content: '';
                position: absolute;
                left: -1.7rem;
                top: 12px;
                width: 2px;
                height: calc(100% - 12px);
                background: #e0e0e0;
            }
            
            .timeline-item:last-child::after {
                display: none;
            }
            
            .timeline-year {
                color: #6366f1;
                font-weight: bold;
                font-size: 1.2rem;
                margin-bottom: 0.5rem;
            }
        </style>
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
                    <a href="../terms-of-service/">Terms of Service</a>
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
                    <p>{{ site_name }} is an innovative AI-powered technology blog dedicated to delivering high-quality, informative content on the latest developments in artificial intelligence, machine learning, and emerging technologies. We leverage cutting-edge AI technology to bring you timely, relevant, and insightful articles that keep you informed about the rapidly evolving tech landscape.</p>
                    <p>Founded with a vision to democratize access to technology knowledge, we combine the power of artificial intelligence with editorial oversight to create content that is both accurate and accessible to our diverse readership.</p>
                </div>

                <div class="mission-box">
                    <h2>🎯 Our Mission</h2>
                    <p>To empower readers with knowledge about artificial intelligence and technology through accessible, accurate, and engaging content that bridges the gap between complex technical concepts and everyday understanding.</p>
                </div>

                <div class="about-section">
                    <h2>📊 What We Do</h2>
                    <p>At {{ site_name }}, we focus on creating content that matters. Our AI-powered platform analyzes trends, research papers, and industry developments to bring you:</p>
                    
                    <div class="feature-grid">
                        <div class="feature-card">
                            <span class="feature-icon">📰</span>
                            <h3>Breaking News</h3>
                            <p>Stay updated with the latest AI and tech announcements, product launches, and industry developments.</p>
                        </div>
                        
                        <div class="feature-card">
                            <span class="feature-icon">🔬</span>
                            <h3>In-Depth Analysis</h3>
                            <p>Comprehensive breakdowns of complex topics, research findings, and emerging technologies.</p>
                        </div>
                        
                        <div class="feature-card">
                            <span class="feature-icon">💡</span>
                            <h3>Practical Guides</h3>
                            <p>Step-by-step tutorials and how-to guides for implementing AI solutions and tools.</p>
                        </div>
                        
                        <div class="feature-card">
                            <span class="feature-icon">🚀</span>
                            <h3>Future Trends</h3>
                            <p>Forward-looking analysis of where AI and technology are headed and what it means for you.</p>
                        </div>
                        
                        <div class="feature-card">
                            <span class="feature-icon">⚖️</span>
                            <h3>Ethics & Policy</h3>
                            <p>Thoughtful discussions on AI ethics, regulations, and responsible technology development.</p>
                        </div>
                        
                        <div class="feature-card">
                            <span class="feature-icon">🎓</span>
                            <h3>Educational Content</h3>
                            <p>Learning resources for everyone from beginners to advanced practitioners.</p>
                        </div>
                    </div>
                </div>

                <div class="about-section">
                    <h2>💻 Topics We Cover</h2>
                    <p>Our content spans a wide range of technology topics, with particular focus on:</p>
                    
                    <div class="topics-grid">
                        {% for topic in topics %}
                        <div class="topic-tag">{{ topic }}</div>
                        {% endfor %}
                    </div>
                </div>

                <div class="about-section">
                    <h2>🤖 Our AI-Powered Approach</h2>
                    
                    <h3>Content Creation Philosophy</h3>
                    <p>We believe in the responsible use of AI technology to create content that genuinely helps our readers. Our approach combines:</p>
                    <ul>
                        <li><strong>Advanced AI Models:</strong> We use state-of-the-art language models trained on vast datasets to generate initial content drafts</li>
                        <li><strong>Quality Control:</strong> Every piece of content goes through automated quality checks for accuracy, readability, and relevance</li>
                        <li><strong>Editorial Standards:</strong> We maintain strict editorial guidelines to ensure our content meets high standards</li>
                        <li><strong>Continuous Improvement:</strong> We regularly update our AI systems based on feedback and emerging best practices</li>
                        <li><strong>Transparency:</strong> We're open about our use of AI and clearly disclose AI-generated content</li>
                    </ul>

                    <h3>Quality Assurance</h3>
                    <p>While AI powers our content creation, we implement multiple safeguards:</p>
                    <ul>
                        <li>Fact-checking protocols to verify key claims and statistics</li>
                        <li>Source verification to ensure information comes from reputable sources</li>
                        <li>Readability optimization for diverse audience comprehension levels</li>
                        <li>SEO best practices to help you find the information you need</li>
                        <li>Regular content audits to maintain accuracy and relevance</li>
                    </ul>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">24/7</span>
                        <span class="stat-label">Content Publishing</span>
                    </div>
                    
                    <div class="stat-card">
                        <span class="stat-number">100%</span>
                        <span class="stat-label">AI-Powered</span>
                    </div>
                    
                    <div class="stat-card">
                        <span class="stat-number">∞</span>
                        <span class="stat-label">Learning & Improving</span>
                    </div>
                </div>

                <div class="team-section">
                    <h2>👥 Behind the Technology</h2>
                    <p>{{ site_name }} is operated by a team passionate about artificial intelligence and its potential to transform how we create and consume content. We combine expertise in:</p>
                    <ul>
                        <li><strong>Machine Learning Engineering:</strong> Building and maintaining AI systems that power our content</li>
                        <li><strong>Content Strategy:</strong> Ensuring our articles meet reader needs and interests</li>
                        <li><strong>Technology Journalism:</strong> Following best practices in tech reporting and analysis</li>
                        <li><strong>Web Development:</strong> Creating a seamless user experience</li>
                        <li><strong>SEO & Analytics:</strong> Helping readers discover and engage with our content</li>
                    </ul>
                </div>

                <div class="about-section">
                    <h2>🌍 Our Values</h2>
                    
                    <h3>Transparency</h3>
                    <p>We're honest about our use of AI technology and clearly disclose when content is AI-generated. We believe transparency builds trust.</p>
                    
                    <h3>Accuracy</h3>
                    <p>We strive for factual accuracy in all our content and encourage readers to verify important information from multiple sources.</p>
                    
                    <h3>Accessibility</h3>
                    <p>We make complex technology topics accessible to readers of all backgrounds, from beginners to experts.</p>
                    
                    <h3>Responsibility</h3>
                    <p>We take seriously our responsibility to provide accurate information and acknowledge the limitations of AI-generated content.</p>
                    
                    <h3>Innovation</h3>
                    <p>We continuously explore new ways to improve our content quality and reader experience through technological advancement.</p>
                    
                    <h3>Community</h3>
                    <p>We value our readers' feedback and engagement, using it to improve and evolve our content offerings.</p>
                </div>

                <div class="about-section">
                    <h2>📅 Our Journey</h2>
                    <div class="timeline">
                        <div class="timeline-item">
                            <div class="timeline-year">2025</div>
                            <h3>Foundation</h3>
                            <p>{{ site_name }} launched with a vision to leverage AI for creating accessible technology content.</p>
                        </div>
                        
                        <div class="timeline-item">
                            <div class="timeline-year">Present</div>
                            <h3>Continuous Evolution</h3>
                            <p>We're constantly improving our AI systems, expanding our topic coverage, and enhancing the reader experience.</p>
                        </div>
                        
                        <div class="timeline-item">
                            <div class="timeline-year">Future</div>
                            <h3>Expanding Horizons</h3>
                            <p>Plans for multimedia content, interactive features, and deeper community engagement.</p>
                        </div>
                    </div>
                </div>

                <div class="about-section">
                    <h2>🤝 Partnerships & Collaborations</h2>
                    <p>We're open to collaborating with:</p>
                    <ul>
                        <li><strong>Technology Companies:</strong> For product reviews, case studies, and sponsored content</li>
                        <li><strong>Research Institutions:</strong> To share cutting-edge research findings with our audience</li>
                        <li><strong>Industry Experts:</strong> For guest contributions and expert insights</li>
                        <li><strong>Educational Organizations:</strong> To provide learning resources and educational content</li>
                        <li><strong>Media Outlets:</strong> For content syndication and cross-promotion</li>
                    </ul>
                    <p>Interested in partnering with us? <a href="../contact/">Get in touch</a> to discuss opportunities.</p>
                </div>

                <div class="about-section">
                    <h2>⚖️ Ethical Standards</h2>
                    
                    <h3>Content Integrity</h3>
                    <p>We maintain clear editorial standards and never compromise content quality for commercial interests.</p>
                    
                    <h3>Disclosure</h3>
                    <p>We clearly disclose:</p>
                    <ul>
                        <li>When content is AI-generated</li>
                        <li>Affiliate relationships and sponsored content</li>
                        <li>Potential conflicts of interest</li>
                        <li>Limitations of the information provided</li>
                    </ul>
                    
                    <h3>Privacy</h3>
                    <p>We respect your privacy and handle your data responsibly. See our <a href="../privacy-policy/">Privacy Policy</a> for details.</p>
                    
                    <h3>Attribution</h3>
                    <p>We credit sources appropriately and respect intellectual property rights.</p>
                </div>

                <div class="about-section">
                    <h2>📚 Content Categories</h2>
                    <p>Our content is organized into several key categories to help you find what you're looking for:</p>
                    <ul>
                        <li><strong>News & Updates:</strong> Latest developments in AI and technology</li>
                        <li><strong>Tutorials & Guides:</strong> Practical how-to content for implementing solutions</li>
                        <li><strong>Analysis & Opinion:</strong> Deep dives into trends and implications</li>
                        <li><strong>Research Summaries:</strong> Breaking down academic papers and studies</li>
                        <li><strong>Product Reviews:</strong> Evaluations of AI tools and platforms</li>
                        <li><strong>Career & Education:</strong> Resources for AI and tech professionals</li>
                    </ul>
                </div>

                <div class="cta-box">
                    <h3>📬 Stay Connected</h3>
                    <p>Want to stay updated with the latest AI and technology insights? Get in touch with us!</p>
                    <a href="../contact/" class="cta-button">Contact Us</a>
                </div>

                <div class="about-section">
                    <h2>❓ Frequently Asked Questions</h2>
                    
                    <h3>Is all your content AI-generated?</h3>
                    <p>Yes, the majority of our content is created using AI technology. We're transparent about this and implement quality controls to ensure accuracy and usefulness.</p>
                    
                    <h3>How accurate is AI-generated content?</h3>
                    <p>While we strive for accuracy, AI-generated content can contain errors. We encourage readers to verify critical information and use our content as a starting point for further research.</p>
                    
                    <h3>Can I republish your content?</h3>
                    <p>Please contact us for permission. We're open to content partnerships with proper attribution.</p>
                    
                    <h3>Do you accept guest posts?</h3>
                    <p>Yes! We welcome high-quality guest contributions from industry experts. See our <a href="../contact/">contact page</a> for submission guidelines.</p>
                    
                    <h3>How do you monetize the site?</h3>
                    <p>We use display advertising, affiliate marketing, and sponsored content to support our operations. All commercial relationships are clearly disclosed.</p>
                    
                    <h3>How can I report an error?</h3>
                    <p>We appreciate corrections! Please <a href="../contact/">contact us</a> with details about any inaccuracies you find.</p>
                </div>

                <div class="about-section">
                    <h2>🔮 Looking Ahead</h2>
                    <p>The future of AI and technology is bright, and we're excited to be part of the journey. Our roadmap includes:</p>
                    <ul>
                        <li>Expanding into video and podcast content</li>
                        <li>Developing interactive learning tools and resources</li>
                        <li>Building a community forum for discussions</li>
                        <li>Launching specialized series on emerging tech topics</li>
                        <li>Partnering with more industry leaders and experts</li>
                        <li>Improving personalization and content recommendations</li>
                    </ul>
                    <p>We're committed to evolving with the technology landscape and continuing to serve our readers with valuable, timely content.</p>
                </div>

                <div class="mission-box">
                    <h2>🙏 Thank You</h2>
                    <p>Thank you for being part of the {{ site_name }} community. Your readership and engagement inspire us to continuously improve and deliver the best possible content. Together, we're exploring the fascinating world of artificial intelligence and technology.</p>
                </div>
            </article>
        </main>
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
            </div>
        </footer>
        <!-- Enhanced Navigation Script -->
        <script src="../static/navigation.js"></script>
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
        <style>
            .privacy-section {
                background: #f8f9fa;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #6366f1;
            }
            
            .privacy-section h3 {
                color: #333;
                margin-top: 0;
                margin-bottom: 1rem;
                font-size: 1.3rem;
            }
            
            .privacy-section h4 {
                color: #555;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
                font-size: 1.05rem;
            }
            
            .privacy-section ul {
                margin-left: 1.5rem;
                line-height: 1.8;
            }
            
            .privacy-section ul li {
                margin-bottom: 0.5rem;
            }
            
            .highlight-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 2rem 0;
            }
            
            .highlight-box h3 {
                margin-top: 0;
                color: white;
            }
            
            .important-notice {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 1rem 1.5rem;
                margin: 1.5rem 0;
                border-radius: 4px;
            }
            
            .table-container {
                overflow-x: auto;
                margin: 1.5rem 0;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
            }
            
            th, td {
                padding: 1rem;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            
            th {
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }
            
            .toc {
                background: #f0f4ff;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 2rem;
            }
            
            .toc h3 {
                margin-top: 0;
                color: #333;
            }
            
            .toc ul {
                list-style: none;
                padding-left: 0;
            }
            
            .toc li {
                margin-bottom: 0.5rem;
            }
            
            .toc a {
                color: #6366f1;
                text-decoration: none;
            }
            
            .toc a:hover {
                text-decoration: underline;
            }
        </style>
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
                    <a href="../terms-of-service/">Terms of Service</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>Privacy Policy</h2>
                <p>How we protect and handle your information</p>
            </div>
            <article class="page-content">
                <div class="toc">
                    <h3>Table of Contents</h3>
                    <ul>
                        <li><a href="#introduction">1. Introduction</a></li>
                        <li><a href="#information-collection">2. Information We Collect</a></li>
                        <li><a href="#how-we-use">3. How We Use Your Information</a></li>
                        <li><a href="#cookies">4. Cookies and Tracking Technologies</a></li>
                        <li><a href="#third-party">5. Third-Party Services</a></li>
                        <li><a href="#data-sharing">6. Data Sharing and Disclosure</a></li>
                        <li><a href="#your-rights">7. Your Privacy Rights</a></li>
                        <li><a href="#data-retention">8. Data Retention</a></li>
                        <li><a href="#security">9. Security Measures</a></li>
                        <li><a href="#children">10. Children's Privacy</a></li>
                        <li><a href="#international">11. International Data Transfers</a></li>
                        <li><a href="#changes">12. Changes to This Policy</a></li>
                        <li><a href="#contact">13. Contact Information</a></li>
                    </ul>
                </div>

                <div id="introduction" class="privacy-section">
                    <h3>1. Introduction</h3>
                    <p><strong>AI Tech Blog</strong> ("we," "our," "us," or the "Site") is committed to protecting your privacy and ensuring transparency in how we handle your personal information. This Privacy Policy explains our practices regarding the collection, use, storage, and protection of your data when you visit or interact with our website.</p>
                    <p>By accessing or using our Site, you acknowledge that you have read, understood, and agree to be bound by this Privacy Policy. If you do not agree with our practices, please do not use our Site.</p>
                </div>

                <div id="information-collection" class="privacy-section">
                    <h3>2. Information We Collect</h3>
                    
                    <h4>2.1 Information You Provide Directly</h4>
                    <ul>
                        <li><strong>Contact Information:</strong> When you contact us via email or contact forms, we collect your name, email address, and any other information you choose to provide in your message.</li>
                        <li><strong>Newsletter Subscriptions:</strong> If you subscribe to our newsletter, we collect your email address and subscription preferences.</li>
                        <li><strong>Comments and Feedback:</strong> Any comments, suggestions, or feedback you provide through our Site.</li>
                    </ul>

                    <h4>2.2 Information Collected Automatically</h4>
                    <ul>
                        <li><strong>Device Information:</strong> Device type, operating system, browser type and version, screen resolution</li>
                        <li><strong>Usage Data:</strong> Pages visited, time spent on pages, links clicked, referring/exit pages</li>
                        <li><strong>Location Data:</strong> General geographic location (country, region, city) based on IP address</li>
                        <li><strong>Log Data:</strong> IP address, access times, browser information, and crash reports</li>
                    </ul>

                    <h4>2.3 Information from Third Parties</h4>
                    <ul>
                        <li><strong>Analytics Providers:</strong> We receive aggregated data from services like Google Analytics</li>
                        <li><strong>Advertising Networks:</strong> Information about ad impressions and interactions</li>
                        <li><strong>Social Media Platforms:</strong> If you interact with our social media content</li>
                    </ul>
                </div>

                <div id="how-we-use" class="privacy-section">
                    <h3>3. How We Use Your Information</h3>
                    <p>We use the collected information for the following purposes:</p>

                    <h4>3.1 Site Operations and Maintenance</h4>
                    <ul>
                        <li>Operate, maintain, and improve the Site and its features</li>
                        <li>Process and respond to your inquiries and requests</li>
                        <li>Monitor and analyze usage trends and preferences</li>
                        <li>Detect, prevent, and address technical issues</li>
                    </ul>

                    <h4>3.2 Content and Personalization</h4>
                    <ul>
                        <li>Deliver relevant and personalized content</li>
                        <li>Understand which content resonates with our audience</li>
                        <li>Improve our AI-generated content quality</li>
                        <li>Recommend articles and topics based on interests</li>
                    </ul>

                    <h4>3.3 Communications</h4>
                    <ul>
                        <li>Send newsletters and updates (with your consent)</li>
                        <li>Respond to comments, questions, and support requests</li>
                        <li>Send important notices about changes to our policies</li>
                        <li>Notify you about new features or content</li>
                    </ul>

                    <h4>3.4 Analytics and Research</h4>
                    <ul>
                        <li>Analyze site performance and user behavior</li>
                        <li>Conduct research to improve our services</li>
                        <li>Generate statistical reports and insights</li>
                    </ul>

                    <h4>3.5 Legal and Compliance</h4>
                    <ul>
                        <li>Comply with legal obligations and regulatory requirements</li>
                        <li>Protect our rights, privacy, safety, or property</li>
                        <li>Enforce our Terms of Service</li>
                        <li>Respond to legal requests from authorities</li>
                    </ul>
                </div>

                <div id="cookies" class="privacy-section">
                    <h3>4. Cookies and Tracking Technologies</h3>
                    
                    <h4>4.1 What Are Cookies?</h4>
                    <p>Cookies are small text files stored on your device that help us improve your experience, understand how you use our Site, and deliver relevant content and advertisements.</p>

                    <h4>4.2 Types of Cookies We Use</h4>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Cookie Type</th>
                                    <th>Purpose</th>
                                    <th>Duration</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Essential Cookies</strong></td>
                                    <td>Required for basic site functionality</td>
                                    <td>Session</td>
                                </tr>
                                <tr>
                                    <td><strong>Analytics Cookies</strong></td>
                                    <td>Track site usage and performance</td>
                                    <td>Up to 2 years</td>
                                </tr>
                                <tr>
                                    <td><strong>Advertising Cookies</strong></td>
                                    <td>Deliver relevant advertisements</td>
                                    <td>Up to 1 year</td>
                                </tr>
                                <tr>
                                    <td><strong>Preference Cookies</strong></td>
                                    <td>Remember your settings and preferences</td>
                                    <td>Up to 1 year</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <h4>4.3 Managing Cookies</h4>
                    <p>You can control cookies through your browser settings. However, disabling certain cookies may limit your ability to use some features of our Site. Most browsers accept cookies automatically, but you can modify your browser settings to decline cookies if you prefer.</p>

                    <h4>4.4 Other Tracking Technologies</h4>
                    <ul>
                        <li><strong>Web Beacons:</strong> Small graphic images used in emails and web pages to track engagement</li>
                        <li><strong>Pixel Tags:</strong> Used to measure advertising campaign effectiveness</li>
                        <li><strong>Local Storage:</strong> Browser storage used to save preferences and settings</li>
                    </ul>
                </div>

                <div id="third-party" class="privacy-section">
                    <h3>5. Third-Party Services</h3>
                    
                    <h4>5.1 Analytics Services</h4>
                    <ul>
                        <li><strong>Google Analytics:</strong> We use Google Analytics to understand how visitors interact with our Site. Google Analytics collects information anonymously and reports website trends without identifying individual visitors. <a href="https://policies.google.com/privacy" target="_blank" rel="noopener">Google Privacy Policy</a></li>
                    </ul>

                    <h4>5.2 Advertising Networks</h4>
                    <ul>
                        <li><strong>Google AdSense:</strong> We use Google AdSense to display advertisements on our Site. Google may use cookies to serve ads based on your prior visits to our Site or other websites. <a href="https://policies.google.com/technologies/ads" target="_blank" rel="noopener">Google Advertising Policy</a></li>
                        <li>You can opt out of personalized advertising by visiting <a href="https://www.aboutads.info/choices/" target="_blank" rel="noopener">aboutads.info</a></li>
                    </ul>

                    <h4>5.3 Hosting and Infrastructure</h4>
                    <ul>
                        <li>We use third-party hosting providers to store and process data</li>
                        <li>These providers are contractually obligated to protect your data</li>
                    </ul>

                    <h4>5.4 Affiliate Programs</h4>
                    <p>We participate in affiliate marketing programs. When you click on an affiliate link and make a purchase, we may receive a commission. The affiliate partner may collect information about your interaction with their site according to their own privacy policy.</p>
                </div>

                <div id="data-sharing" class="privacy-section">
                    <h3>6. Data Sharing and Disclosure</h3>
                    
                    <div class="important-notice">
                        <p><strong>Important:</strong> We do not sell, rent, or trade your personal information to third parties for marketing purposes.</p>
                    </div>

                    <p>We may share your information in the following circumstances:</p>

                    <h4>6.1 Service Providers</h4>
                    <p>We share information with trusted service providers who assist us in operating our Site, conducting our business, or serving our users, such as:</p>
                    <ul>
                        <li>Web hosting providers</li>
                        <li>Analytics services</li>
                        <li>Email service providers</li>
                        <li>Content delivery networks</li>
                    </ul>

                    <h4>6.2 Legal Requirements</h4>
                    <p>We may disclose your information if required by law or in response to:</p>
                    <ul>
                        <li>Court orders, subpoenas, or legal processes</li>
                        <li>Requests from government authorities</li>
                        <li>Investigation of potential violations of our policies</li>
                        <li>Protection of our rights, safety, or property</li>
                    </ul>

                    <h4>6.3 Business Transfers</h4>
                    <p>In the event of a merger, acquisition, reorganization, or sale of assets, your information may be transferred as part of that transaction.</p>

                    <h4>6.4 Aggregated or Anonymized Data</h4>
                    <p>We may share aggregated or anonymized data that cannot reasonably be used to identify you with third parties for analytics, research, or marketing purposes.</p>
                </div>

                <div id="your-rights" class="privacy-section">
                    <h3>7. Your Privacy Rights</h3>
                    
                    <p>Depending on your location, you may have the following rights regarding your personal data:</p>

                    <h4>7.1 General Rights</h4>
                    <ul>
                        <li><strong>Access:</strong> Request a copy of the personal information we hold about you</li>
                        <li><strong>Correction:</strong> Request correction of inaccurate or incomplete information</li>
                        <li><strong>Deletion:</strong> Request deletion of your personal information (subject to legal obligations)</li>
                        <li><strong>Objection:</strong> Object to processing of your personal information</li>
                        <li><strong>Restriction:</strong> Request restriction of processing in certain circumstances</li>
                        <li><strong>Portability:</strong> Request transfer of your data to another service</li>
                        <li><strong>Withdraw Consent:</strong> Withdraw consent at any time (where processing is based on consent)</li>
                    </ul>

                    <h4>7.2 GDPR Rights (EU/EEA Residents)</h4>
                    <p>If you are in the European Union or European Economic Area, you have additional rights under the General Data Protection Regulation (GDPR), including the right to lodge a complaint with your local data protection authority.</p>

                    <h4>7.3 CCPA Rights (California Residents)</h4>
                    <p>If you are a California resident, you have rights under the California Consumer Privacy Act (CCPA), including:</p>
                    <ul>
                        <li>Right to know what personal information is collected</li>
                        <li>Right to know if personal information is sold or disclosed</li>
                        <li>Right to opt-out of the sale of personal information</li>
                        <li>Right to deletion</li>
                        <li>Right to non-discrimination for exercising your rights</li>
                    </ul>

                    <h4>7.4 How to Exercise Your Rights</h4>
                    <p>To exercise any of these rights, please contact us at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>. We will respond to your request within 30 days (or as required by applicable law).</p>
                </div>

                <div id="data-retention" class="privacy-section">
                    <h3>8. Data Retention</h3>
                    
                    <p>We retain your personal information for as long as necessary to fulfill the purposes outlined in this Privacy Policy, unless a longer retention period is required or permitted by law.</p>

                    <h4>8.1 Retention Periods</h4>
                    <ul>
                        <li><strong>Contact Information:</strong> Retained until you request deletion or unsubscribe</li>
                        <li><strong>Analytics Data:</strong> Typically retained for 26 months (Google Analytics default)</li>
                        <li><strong>Log Data:</strong> Automatically deleted after 12 months</li>
                        <li><strong>Cookies:</strong> Expire according to the durations specified in Section 4.2</li>
                    </ul>

                    <h4>8.2 Deletion Criteria</h4>
                    <p>When determining retention periods, we consider:</p>
                    <ul>
                        <li>The purpose for which we collected the information</li>
                        <li>Legal and regulatory requirements</li>
                        <li>The sensitivity of the information</li>
                        <li>Our legitimate business interests</li>
                    </ul>
                </div>

                <div id="security" class="privacy-section">
                    <h3>9. Security Measures</h3>
                    
                    <p>We implement appropriate technical and organizational measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction.</p>

                    <h4>9.1 Security Practices</h4>
                    <ul>
                        <li><strong>Encryption:</strong> Data transmission is protected using SSL/TLS encryption</li>
                        <li><strong>Access Controls:</strong> Limited access to personal information on a need-to-know basis</li>
                        <li><strong>Regular Monitoring:</strong> Continuous monitoring for security vulnerabilities</li>
                        <li><strong>Secure Storage:</strong> Data stored on secure servers with appropriate safeguards</li>
                        <li><strong>Regular Updates:</strong> Systems and software regularly updated and patched</li>
                    </ul>

                    <h4>9.2 Important Security Notice</h4>
                    <div class="important-notice">
                        <p><strong>Please Note:</strong> While we implement reasonable security measures, no method of transmission over the Internet or electronic storage is 100% secure. We cannot guarantee absolute security of your information.</p>
                    </div>

                    <h4>9.3 Your Responsibility</h4>
                    <p>You are responsible for:</p>
                    <ul>
                        <li>Keeping your contact information secure</li>
                        <li>Using strong passwords if you create an account</li>
                        <li>Notifying us immediately of any unauthorized access</li>
                    </ul>
                </div>

                <div id="children" class="privacy-section">
                    <h3>10. Children's Privacy</h3>
                    
                    <p>Our Site is not intended for children under the age of 13 (or the age of digital consent in your jurisdiction). We do not knowingly collect personal information from children.</p>

                    <h4>10.1 If You Are a Parent or Guardian</h4>
                    <p>If you believe that your child has provided us with personal information without your consent, please contact us immediately at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>. We will take steps to delete such information promptly.</p>

                    <h4>10.2 Age Verification</h4>
                    <p>We do not require age verification for general Site browsing. However, we recommend that parents and guardians monitor their children's online activities.</p>
                </div>

                <div id="international" class="privacy-section">
                    <h3>11. International Data Transfers</h3>
                    
                    <p>Your information may be transferred to and processed in countries other than your country of residence. These countries may have data protection laws that differ from those in your jurisdiction.</p>

                    <h4>11.1 Safeguards</h4>
                    <p>When we transfer your information internationally, we ensure appropriate safeguards are in place, such as:</p>
                    <ul>
                        <li>Standard contractual clauses approved by relevant authorities</li>
                        <li>Adequacy decisions by data protection authorities</li>
                        <li>Participation in recognized privacy frameworks</li>
                    </ul>

                    <h4>11.2 Your Location</h4>
                    <p>Our primary operations are based in Kenya. By using our Site, you consent to the transfer of your information to Kenya and other jurisdictions where we or our service providers operate.</p>
                </div>

                <div id="changes" class="privacy-section">
                    <h3>12. Changes to This Privacy Policy</h3>
                    
                    <p>We may update this Privacy Policy from time to time to reflect changes in our practices, technology, legal requirements, or other factors.</p>

                    <h4>12.1 Notification of Changes</h4>
                    <ul>
                        <li>Material changes will be posted prominently on our Site</li>
                        <li>The "Last Updated" date at the top will be revised</li>
                        <li>For significant changes, we may send email notifications to registered users</li>
                        <li>We encourage you to review this policy periodically</li>
                    </ul>

                    <h4>12.2 Continued Use</h4>
                    <p>Your continued use of the Site after changes to this Privacy Policy constitutes your acceptance of the updated policy.</p>
                </div>

                <div id="contact" class="highlight-box">
                    <h3>13. Contact Information</h3>
                    
                    <p>If you have any questions, concerns, or requests regarding this Privacy Policy or our data practices, please contact us:</p>
                    
                    <p><strong>Email:</strong> <a href="mailto:aiblogauto@gmail.com" style="color: white; text-decoration: underline;">aiblogauto@gmail.com</a></p>
                    
                    <p><strong>Response Time:</strong> We aim to respond to all inquiries within 48 hours during business days.</p>
                    
                    <p><strong>Data Protection Officer:</strong> For privacy-related inquiries, please include "Privacy Request" in your email subject line.</p>
                </div>

                <div class="important-notice">
                    <h4>Important Legal Notice</h4>
                    <p>This Privacy Policy is part of our Terms of Service. By using our Site, you agree to both documents. If there is any conflict between this Privacy Policy and our Terms of Service, the Terms of Service shall prevail.</p>
                </div>

                <div style="margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <p><strong>Last Updated:</strong> January 28, 2026</p>
                </div>
            </article>
        </main>

        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
            </div>
        </footer>
         <!-- Enhanced Navigation Script -->
        <script src="../static/navigation.js"></script>
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
        <style>
            .terms-section {
                background: #f8f9fa;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #6366f1;
            }
            
            .terms-section h3 {
                color: #333;
                margin-top: 0;
                margin-bottom: 1rem;
                font-size: 1.3rem;
            }
            
            .terms-section h4 {
                color: #555;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
                font-size: 1.05rem;
            }
            
            .terms-section ul, .terms-section ol {
                margin-left: 1.5rem;
                line-height: 1.8;
            }
            
            .terms-section ul li, .terms-section ol li {
                margin-bottom: 0.5rem;
            }
            
            .highlight-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 2rem 0;
            }
            
            .highlight-box h3 {
                margin-top: 0;
                color: white;
            }
            
            .important-notice {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 1rem 1.5rem;
                margin: 1.5rem 0;
                border-radius: 4px;
            }
            
            .warning-box {
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 1rem 1.5rem;
                margin: 1.5rem 0;
                border-radius: 4px;
                color: #721c24;
            }
            
            .toc {
                background: #f0f4ff;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 2rem;
            }
            
            .toc h3 {
                margin-top: 0;
                color: #333;
            }
            
            .toc ul {
                list-style: none;
                padding-left: 0;
            }
            
            .toc li {
                margin-bottom: 0.5rem;
            }
            
            .toc a {
                color: #6366f1;
                text-decoration: none;
            }
            
            .toc a:hover {
                text-decoration: underline;
            }
            
            .definitions-box {
                background: #e7f3ff;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #0066cc;
                margin: 1.5rem 0;
            }
        </style>
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
                    <a href="../terms-of-service/">Terms of Service</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>Terms of Service</h2>
                <p>Please read these terms carefully before using our site</p>
            </div>
            <article class="page-content">
                <div class="important-notice">
                    <p><strong>Important:</strong> These Terms of Service constitute a legally binding agreement between you and AI Tech Blog. By accessing or using our Site, you agree to be bound by these Terms. If you do not agree with any part of these Terms, you must not use our Site.</p>
                </div>

                <div class="toc">
                    <h3>Table of Contents</h3>
                    <ul>
                        <li><a href="#acceptance">1. Acceptance of Terms</a></li>
                        <li><a href="#definitions">2. Definitions</a></li>
                        <li><a href="#eligibility">3. Eligibility and User Requirements</a></li>
                        <li><a href="#use-license">4. License to Use Site</a></li>
                        <li><a href="#ai-content">5. AI-Generated Content Disclaimer</a></li>
                        <li><a href="#user-conduct">6. User Conduct and Prohibited Activities</a></li>
                        <li><a href="#intellectual-property">7. Intellectual Property Rights</a></li>
                        <li><a href="#user-content">8. User-Generated Content</a></li>
                        <li><a href="#third-party">9. Third-Party Links and Affiliate Disclosure</a></li>
                        <li><a href="#disclaimer">10. Disclaimers and Warranties</a></li>
                        <li><a href="#limitation">11. Limitation of Liability</a></li>
                        <li><a href="#indemnification">12. Indemnification</a></li>
                        <li><a href="#privacy">13. Privacy Policy</a></li>
                        <li><a href="#dmca">14. DMCA and Copyright Infringement</a></li>
                        <li><a href="#termination">15. Termination and Suspension</a></li>
                        <li><a href="#modifications">16. Modifications to Terms</a></li>
                        <li><a href="#dispute">17. Dispute Resolution</a></li>
                        <li><a href="#governing-law">18. Governing Law and Jurisdiction</a></li>
                        <li><a href="#miscellaneous">19. Miscellaneous Provisions</a></li>
                        <li><a href="#contact">20. Contact Information</a></li>
                    </ul>
                </div>

                <div id="acceptance" class="terms-section">
                    <h3>1. Acceptance of Terms</h3>
                    <p>By accessing, browsing, or using AI Tech Blog (the "Site," "Service," "we," "our," or "us"), you acknowledge that you have read, understood, and agree to be bound by these Terms of Service (the "Terms") and our Privacy Policy.</p>
                    
                    <h4>1.1 Agreement Formation</h4>
                    <p>These Terms form a binding legal agreement between you (the "User," "you," or "your") and AI Tech Blog. Your use of the Site constitutes acceptance of these Terms.</p>
                    
                    <h4>1.2 Modifications</h4>
                    <p>We reserve the right to modify these Terms at any time. Continued use of the Site after changes constitutes acceptance of the modified Terms.</p>
                </div>

                <div id="definitions" class="definitions-box">
                    <h3>2. Definitions</h3>
                    <p>For the purposes of these Terms:</p>
                    <ul>
                        <li><strong>"Site"</strong> refers to AI Tech Blog, including all web pages, content, features, and services available at our domain.</li>
                        <li><strong>"User" or "you"</strong> refers to any individual or entity accessing or using the Site.</li>
                        <li><strong>"Content"</strong> refers to all text, images, videos, audio, code, data, and other materials on the Site.</li>
                        <li><strong>"AI-Generated Content"</strong> refers to articles, posts, and materials created using artificial intelligence technology.</li>
                        <li><strong>"User Content"</strong> refers to any content submitted, posted, or transmitted by Users.</li>
                        <li><strong>"Services"</strong> refers to all features, functionality, and services provided through the Site.</li>
                        <li><strong>"Third-Party"</strong> refers to any person or entity other than you or AI Tech Blog.</li>
                    </ul>
                </div>

                <div id="eligibility" class="terms-section">
                    <h3>3. Eligibility and User Requirements</h3>
                    
                    <h4>3.1 Age Requirements</h4>
                    <p>You must be at least 13 years old to use this Site. If you are between 13 and 18 years old (or the age of legal majority in your jurisdiction), you may only use the Site with the consent and supervision of a parent or legal guardian.</p>
                    
                    <h4>3.2 Legal Capacity</h4>
                    <p>By using the Site, you represent and warrant that:</p>
                    <ul>
                        <li>You have the legal capacity to enter into binding agreements</li>
                        <li>You are not prohibited from using the Site under applicable laws</li>
                        <li>All information you provide is accurate and current</li>
                        <li>You will maintain the accuracy of your information</li>
                    </ul>
                    
                    <h4>3.3 Geographic Restrictions</h4>
                    <p>The Site is operated from Kenya. We make no representation that the Site or its Content is appropriate or available for use in all locations. Access to the Site from territories where its Content is illegal is prohibited.</p>
                </div>

                <div id="use-license" class="terms-section">
                    <h3>4. License to Use Site</h3>
                    
                    <h4>4.1 Limited License Grant</h4>
                    <p>Subject to your compliance with these Terms, we grant you a limited, non-exclusive, non-transferable, non-sublicensable, revocable license to:</p>
                    <ul>
                        <li>Access and view the Site and its Content</li>
                        <li>Download and print individual articles for personal, non-commercial use</li>
                        <li>Share links to our Content through social media and other channels</li>
                    </ul>
                    
                    <h4>4.2 Restrictions on Use</h4>
                    <p>You may NOT:</p>
                    <ul>
                        <li>Use the Site for any commercial purpose without our express written consent</li>
                        <li>Modify, copy, distribute, transmit, display, reproduce, or create derivative works from the Site</li>
                        <li>Use automated systems (bots, scrapers, spiders) to access the Site without permission</li>
                        <li>Attempt to gain unauthorized access to any portion of the Site</li>
                        <li>Remove or alter any copyright, trademark, or proprietary notices</li>
                        <li>Frame or mirror any part of the Site without our prior written authorization</li>
                    </ul>
                    
                    <h4>4.3 Permitted Uses</h4>
                    <p>You may:</p>
                    <ul>
                        <li>Quote brief excerpts from our Content with proper attribution</li>
                        <li>Share articles via social media or email using provided sharing features</li>
                        <li>Bookmark or save articles for personal reference</li>
                        <li>Print articles for personal, educational, or non-commercial purposes</li>
                    </ul>
                </div>

                <div id="ai-content" class="terms-section">
                    <h3>5. AI-Generated Content Disclaimer</h3>
                    
                    <div class="warning-box">
                        <p><strong>Critical Notice:</strong> The majority of articles, posts, and recommendations published on AI Tech Blog are automatically generated using artificial intelligence technology.</p>
                    </div>
                    
                    <h4>5.1 Nature of AI Content</h4>
                    <p>Our AI-generated content is created by machine learning models trained on diverse datasets. While we strive for accuracy and quality, AI-generated content may contain:</p>
                    <ul>
                        <li>Factual inaccuracies or errors</li>
                        <li>Outdated information</li>
                        <li>Incomplete analysis or conclusions</li>
                        <li>Biases present in training data</li>
                        <li>Contextual misunderstandings</li>
                    </ul>
                    
                    <h4>5.2 No Professional Advice</h4>
                    <p>Content on this Site should NOT be considered as:</p>
                    <ul>
                        <li><strong>Legal Advice:</strong> Do not rely on our content for legal decisions</li>
                        <li><strong>Financial Advice:</strong> Not a substitute for professional financial consultation</li>
                        <li><strong>Medical Advice:</strong> Never replace professional medical advice with our content</li>
                        <li><strong>Professional Consultation:</strong> Always consult qualified professionals for specific advice</li>
                    </ul>
                    
                    <h4>5.3 User Responsibility</h4>
                    <p>You acknowledge and agree that:</p>
                    <ul>
                        <li>You are responsible for verifying all information before acting on it</li>
                        <li>You use the information at your own risk</li>
                        <li>We are not liable for decisions made based on our content</li>
                        <li>You should conduct independent research and seek professional advice when needed</li>
                    </ul>
                    
                    <h4>5.4 Quality Control</h4>
                    <p>While we implement review processes and quality controls, we cannot guarantee that all AI-generated content is accurate, complete, or up-to-date. We encourage users to report inaccuracies to help us improve.</p>
                </div>

                <div id="user-conduct" class="terms-section">
                    <h3>6. User Conduct and Prohibited Activities</h3>
                    
                    <h4>6.1 Acceptable Use</h4>
                    <p>You agree to use the Site only for lawful purposes and in accordance with these Terms. You agree not to use the Site:</p>
                    <ul>
                        <li>In any way that violates any applicable law or regulation</li>
                        <li>To harm, threaten, or harass other users or third parties</li>
                        <li>To transmit any harmful code, viruses, or malicious software</li>
                        <li>To impersonate any person or entity</li>
                        <li>To collect personal information of other users without consent</li>
                    </ul>
                    
                    <h4>6.2 Prohibited Activities</h4>
                    <p>Specifically prohibited activities include, but are not limited to:</p>
                    <ul>
                        <li><strong>Automated Access:</strong> Using bots, scrapers, or automated tools without authorization</li>
                        <li><strong>Security Violations:</strong> Attempting to breach or test security measures</li>
                        <li><strong>System Interference:</strong> Disrupting or overburdening the Site's infrastructure</li>
                        <li><strong>Unauthorized Access:</strong> Accessing areas of the Site you're not permitted to access</li>
                        <li><strong>Content Theft:</strong> Republishing our content without permission</li>
                        <li><strong>Reverse Engineering:</strong> Attempting to decipher, decompile, or reverse engineer any part of the Site</li>
                        <li><strong>Spam:</strong> Sending unsolicited communications or spam</li>
                        <li><strong>False Information:</strong> Providing false, misleading, or fraudulent information</li>
                    </ul>
                    
                    <h4>6.3 Consequences of Violations</h4>
                    <p>Violations of these conduct rules may result in:</p>
                    <ul>
                        <li>Immediate termination of your access to the Site</li>
                        <li>Legal action, including civil and criminal penalties</li>
                        <li>Cooperation with law enforcement authorities</li>
                        <li>Claims for damages and costs</li>
                    </ul>
                </div>

                <div id="intellectual-property" class="terms-section">
                    <h3>7. Intellectual Property Rights</h3>
                    
                    <h4>7.1 Ownership</h4>
                    <p>All Content on the Site, including but not limited to text, graphics, logos, images, videos, audio, software, and code, is the property of AI Tech Blog or its content suppliers and is protected by:</p>
                    <ul>
                        <li>Copyright laws</li>
                        <li>Trademark laws</li>
                        <li>Patent laws</li>
                        <li>Trade secret laws</li>
                        <li>Other intellectual property rights</li>
                    </ul>
                    
                    <h4>7.2 Trademarks</h4>
                    <p>"AI Tech Blog" and related marks, logos, and designs are trademarks or registered trademarks. You may not use these marks without our prior written permission.</p>
                    
                    <h4>7.3 Copyright Notice</h4>
                    <p>© 2026 AI Tech Blog. All rights reserved. Unauthorized reproduction, distribution, or transmission of Content may result in severe civil and criminal penalties.</p>
                    
                    <h4>7.4 Fair Use</h4>
                    <p>Limited quotation of our content for purposes of commentary, criticism, news reporting, teaching, scholarship, or research may be permitted under fair use doctrine. Always provide proper attribution and link back to the original article.</p>
                </div>

                <div id="user-content" class="terms-section">
                    <h3>8. User-Generated Content</h3>
                    
                    <h4>8.1 Submission of Content</h4>
                    <p>If you submit, post, or transmit any content to the Site (comments, feedback, suggestions, etc.), you grant us:</p>
                    <ul>
                        <li>A worldwide, perpetual, irrevocable, royalty-free license</li>
                        <li>The right to use, reproduce, modify, adapt, publish, and distribute your content</li>
                        <li>The right to create derivative works from your content</li>
                        <li>The right to sublicense these rights to others</li>
                    </ul>
                    
                    <h4>8.2 Content Standards</h4>
                    <p>Any content you submit must:</p>
                    <ul>
                        <li>Not violate any intellectual property rights</li>
                        <li>Not contain unlawful, harmful, or offensive material</li>
                        <li>Not contain viruses or malicious code</li>
                        <li>Not infringe on privacy rights of others</li>
                        <li>Be accurate and not misleading</li>
                    </ul>
                    
                    <h4>8.3 Content Moderation</h4>
                    <p>We reserve the right, but not the obligation, to:</p>
                    <ul>
                        <li>Monitor, review, and remove any user content</li>
                        <li>Edit or modify user content for clarity or compliance</li>
                        <li>Refuse to post or remove content that violates these Terms</li>
                    </ul>
                    
                    <h4>8.4 Representations and Warranties</h4>
                    <p>You represent and warrant that:</p>
                    <ul>
                        <li>You own or have the necessary rights to submit the content</li>
                        <li>Your content does not violate any third-party rights</li>
                        <li>Your content complies with all applicable laws</li>
                    </ul>
                </div>

                <div id="third-party" class="terms-section">
                    <h3>9. Third-Party Links and Affiliate Disclosure</h3>
                    
                    <h4>9.1 Third-Party Links</h4>
                    <p>The Site may contain links to third-party websites, services, or resources. These links are provided for your convenience only.</p>
                    
                    <div class="important-notice">
                        <p><strong>Important:</strong> We do not control, endorse, or assume responsibility for any third-party sites, products, or services. Your interactions with third parties are solely between you and the third party.</p>
                    </div>
                    
                    <h4>9.2 Affiliate Marketing Disclosure</h4>
                    <p>We participate in affiliate marketing programs and may earn commissions from purchases made through affiliate links on our Site. This means:</p>
                    <ul>
                        <li>Some articles contain affiliate links to products or services</li>
                        <li>We receive compensation when you make purchases through these links</li>
                        <li>The price you pay is not affected by our commission</li>
                        <li>We only promote products/services we believe may benefit our readers</li>
                        <li>Affiliate relationships do not influence our editorial content</li>
                    </ul>
                    
                    <h4>9.3 Sponsored Content</h4>
                    <p>We may publish sponsored content or advertisements. Such content will be clearly labeled as "Sponsored," "Advertisement," or similar designations.</p>
                    
                    <h4>9.4 No Warranties for Third-Party Content</h4>
                    <p>We make no warranties or representations about:</p>
                    <ul>
                        <li>The accuracy or reliability of third-party content</li>
                        <li>The quality of third-party products or services</li>
                        <li>The practices or policies of third-party sites</li>
                    </ul>
                </div>

                <div id="disclaimer" class="terms-section">
                    <h3>10. Disclaimers and Warranties</h3>
                    
                    <div class="warning-box">
                        <p><strong>THE SITE AND ALL CONTENT ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND.</strong></p>
                    </div>
                    
                    <h4>10.1 Disclaimer of Warranties</h4>
                    <p>TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE DISCLAIM ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO:</p>
                    <ul>
                        <li>Warranties of merchantability and fitness for a particular purpose</li>
                        <li>Warranties of non-infringement</li>
                        <li>Warranties that the Site will be uninterrupted, error-free, or secure</li>
                        <li>Warranties regarding the accuracy, reliability, or completeness of content</li>
                        <li>Warranties that defects will be corrected</li>
                    </ul>
                    
                    <h4>10.2 No Guarantee of Accuracy</h4>
                    <p>We do not guarantee that:</p>
                    <ul>
                        <li>The Site will meet your requirements or expectations</li>
                        <li>Content is accurate, complete, or current</li>
                        <li>AI-generated content is error-free or reliable</li>
                        <li>The Site will be available at all times</li>
                        <li>Any errors or defects will be corrected</li>
                    </ul>
                    
                    <h4>10.3 Use at Your Own Risk</h4>
                    <p>Your use of the Site and reliance on any content is at your sole risk. You are responsible for any decisions or actions taken based on information obtained from the Site.</p>
                </div>

                <div id="limitation" class="terms-section">
                    <h3>11. Limitation of Liability</h3>
                    
                    <div class="warning-box">
                        <p><strong>IMPORTANT LIMITATION:</strong> TO THE MAXIMUM EXTENT PERMITTED BY LAW, AI TECH BLOG SHALL NOT BE LIABLE FOR ANY DAMAGES ARISING FROM YOUR USE OF THE SITE.</p>
                    </div>
                    
                    <h4>11.1 Exclusion of Damages</h4>
                    <p>IN NO EVENT SHALL WE BE LIABLE FOR ANY:</p>
                    <ul>
                        <li>Direct, indirect, incidental, or consequential damages</li>
                        <li>Loss of profits, revenue, data, or business opportunities</li>
                        <li>Personal injury or property damage</li>
                        <li>Emotional distress or reputational harm</li>
                        <li>Damages resulting from AI-generated content inaccuracies</li>
                        <li>Damages from third-party content, links, or services</li>
                        <li>Damages from unauthorized access or data breaches</li>
                    </ul>
                    
                    <h4>11.2 Cap on Liability</h4>
                    <p>IF ANY LIABILITY IS FOUND TO EXIST DESPITE THESE LIMITATIONS, OUR TOTAL LIABILITY SHALL NOT EXCEED THE AMOUNT YOU PAID TO ACCESS THE SITE (WHICH IS ZERO FOR FREE USERS).</p>
                    
                    <h4>11.3 Basis of the Bargain</h4>
                    <p>You acknowledge that we have set our prices and entered into this agreement in reliance upon these limitations of liability, which allocate risk between us and form the basis of our agreement.</p>
                    
                    <h4>11.4 Jurisdictional Limitations</h4>
                    <p>Some jurisdictions do not allow certain limitations of liability. In such jurisdictions, our liability is limited to the maximum extent permitted by law.</p>
                </div>

                <div id="indemnification" class="terms-section">
                    <h3>12. Indemnification</h3>
                    
                    <h4>12.1 Your Indemnification Obligations</h4>
                    <p>You agree to indemnify, defend, and hold harmless AI Tech Blog, its affiliates, officers, directors, employees, agents, and licensors from and against any claims, liabilities, damages, losses, costs, or expenses (including reasonable attorneys' fees) arising from or related to:</p>
                    <ul>
                        <li>Your use or misuse of the Site</li>
                        <li>Your violation of these Terms</li>
                        <li>Your violation of any law or regulation</li>
                        <li>Your violation of any third-party rights</li>
                        <li>Any content you submit or transmit</li>
                        <li>Your negligent or willful misconduct</li>
                    </ul>
                    
                    <h4>12.2 Defense and Settlement</h4>
                    <p>We reserve the right to assume the exclusive defense and control of any matter subject to indemnification by you, and you agree to cooperate with our defense of such claims.</p>
                </div>

                <div id="privacy" class="terms-section">
                    <h3>13. Privacy Policy</h3>
                    
                    <p>Your use of the Site is also governed by our <a href="../privacy-policy/">Privacy Policy</a>, which is incorporated into these Terms by reference. Please review our Privacy Policy to understand our practices regarding the collection, use, and disclosure of your personal information.</p>
                    
                    <h4>13.1 Data Collection</h4>
                    <p>By using the Site, you consent to:</p>
                    <ul>
                        <li>The collection and use of your information as described in our Privacy Policy</li>
                        <li>The use of cookies and tracking technologies</li>
                        <li>The transfer of your information to our service providers</li>
                    </ul>
                </div>

                <div id="dmca" class="terms-section">
                    <h3>14. DMCA and Copyright Infringement</h3>
                    
                    <h4>14.1 Copyright Policy</h4>
                    <p>We respect the intellectual property rights of others and expect users to do the same. We respond to notices of alleged copyright infringement in accordance with the Digital Millennium Copyright Act (DMCA).</p>
                    
                    <h4>14.2 Filing a DMCA Notice</h4>
                    <p>If you believe that your copyrighted work has been infringed on our Site, please provide our designated agent with the following information:</p>
                    <ul>
                        <li>A physical or electronic signature of the copyright owner or authorized agent</li>
                        <li>Identification of the copyrighted work claimed to be infringed</li>
                        <li>Identification of the material that is claimed to be infringing</li>
                        <li>Your contact information (address, phone number, email)</li>
                        <li>A statement of good faith belief that the use is not authorized</li>
                        <li>A statement under penalty of perjury that the information is accurate</li>
                    </ul>
                    
                    <h4>14.3 Designated DMCA Agent</h4>
                    <p>Copyright notices should be sent to: <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a> with "DMCA Notice" in the subject line.</p>
                    
                    <h4>14.4 Counter-Notification</h4>
                    <p>If you believe that content you posted was wrongly removed, you may file a counter-notification with the required information as specified under DMCA procedures.</p>
                </div>

                <div id="termination" class="terms-section">
                    <h3>15. Termination and Suspension</h3>
                    
                    <h4>15.1 Right to Terminate</h4>
                    <p>We reserve the right to terminate or suspend your access to the Site, without prior notice or liability, for any reason, including but not limited to:</p>
                    <ul>
                        <li>Violation of these Terms</li>
                        <li>Provision of false or misleading information</li>
                        <li>Engagement in fraudulent or illegal activities</li>
                        <li>Behavior that harms other users or the Site</li>
                        <li>At our sole discretion for any other reason</li>
                    </ul>
                    
                    <h4>15.2 Effect of Termination</h4>
                    <p>Upon termination:</p>
                    <ul>
                        <li>Your right to access the Site immediately ceases</li>
                        <li>All provisions of these Terms that should survive termination shall survive</li>
                        <li>We are not liable to you or any third party for termination</li>
                    </ul>
                    
                    <h4>15.3 Survival</h4>
                    <p>The following sections shall survive termination: Intellectual Property Rights, Disclaimers, Limitation of Liability, Indemnification, and Governing Law.</p>
                </div>

                <div id="modifications" class="terms-section">
                    <h3>16. Modifications to Terms and Site</h3>
                    
                    <h4>16.1 Right to Modify Terms</h4>
                    <p>We reserve the right to modify, amend, or update these Terms at any time. Changes will be effective immediately upon posting to the Site.</p>
                    
                    <h4>16.2 Notice of Changes</h4>
                    <p>Material changes will be communicated through:</p>
                    <ul>
                        <li>Prominent notice on the Site</li>
                        <li>Update to the "Last Updated" date</li>
                        <li>Email notification to registered users (if applicable)</li>
                    </ul>
                    
                    <h4>16.3 Acceptance of Modified Terms</h4>
                    <p>Your continued use of the Site after changes constitutes acceptance of the modified Terms. If you do not agree to the changes, you must stop using the Site.</p>
                    
                    <h4>16.4 Right to Modify Site</h4>
                    <p>We reserve the right to modify, suspend, or discontinue any aspect of the Site at any time, with or without notice, and without liability to you.</p>
                </div>

                <div id="dispute" class="terms-section">
                    <h3>17. Dispute Resolution</h3>
                    
                    <h4>17.1 Informal Resolution</h4>
                    <p>Before filing any formal legal action, you agree to first contact us at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a> to attempt to resolve the dispute informally.</p>
                    
                    <h4>17.2 Arbitration Agreement (If Applicable)</h4>
                    <p>If informal resolution is unsuccessful, disputes may be resolved through binding arbitration in accordance with the rules of a recognized arbitration body, to the extent permitted by law.</p>
                    
                    <h4>17.3 Class Action Waiver</h4>
                    <p>TO THE EXTENT PERMITTED BY LAW, YOU AGREE THAT ANY DISPUTE WILL BE RESOLVED INDIVIDUALLY AND YOU WAIVE THE RIGHT TO PARTICIPATE IN A CLASS ACTION.</p>
                    
                    <h4>17.4 Time Limitation</h4>
                    <p>Any claim arising from your use of the Site must be filed within one (1) year after the claim arose. Claims filed after this period are permanently barred.</p>
                </div>

                <div id="governing-law" class="terms-section">
                    <h3>18. Governing Law and Jurisdiction</h3>
                    
                    <h4>18.1 Governing Law</h4>
                    <p>These Terms shall be governed by and construed in accordance with the laws of Kenya, without regard to its conflict of law principles.</p>
                    
                    <h4>18.2 Jurisdiction</h4>
                    <p>You agree to submit to the exclusive jurisdiction of the courts located in Kenya for resolution of any disputes arising from these Terms or your use of the Site.</p>
                    
                    <h4>18.3 International Users</h4>
                    <p>If you access the Site from outside Kenya, you are responsible for compliance with local laws in your jurisdiction.</p>
                </div>

                <div id="miscellaneous" class="terms-section">
                    <h3>19. Miscellaneous Provisions</h3>
                    
                    <h4>19.1 Entire Agreement</h4>
                    <p>These Terms, together with our Privacy Policy, constitute the entire agreement between you and AI Tech Blog regarding the use of the Site.</p>
                    
                    <h4>19.2 Severability</h4>
                    <p>If any provision of these Terms is found to be invalid or unenforceable, the remaining provisions shall remain in full force and effect.</p>
                    
                    <h4>19.3 Waiver</h4>
                    <p>Our failure to enforce any provision of these Terms shall not be deemed a waiver of that provision or of the right to enforce it at a later time.</p>
                    
                    <h4>19.4 Assignment</h4>
                    <p>You may not assign or transfer these Terms or your rights without our prior written consent. We may assign these Terms without restriction.</p>
                    
                    <h4>19.5 Force Majeure</h4>
                    <p>We shall not be liable for any failure or delay in performance due to circumstances beyond our reasonable control.</p>
                    
                    <h4>19.6 Headings</h4>
                    <p>Section headings are for convenience only and do not affect the interpretation of these Terms.</p>
                    
                    <h4>19.7 Language</h4>
                    <p>These Terms are written in English. Any translations are provided for convenience only, and the English version shall control in case of conflicts.</p>
                </div>

                <div id="contact" class="highlight-box">
                    <h3>20. Contact Information</h3>
                    
                    <p>If you have any questions, concerns, or notices regarding these Terms of Service, please contact us:</p>
                    
                    <p><strong>Email:</strong> <a href="mailto:aiblogauto@gmail.com" style="color: white; text-decoration: underline;">aiblogauto@gmail.com</a></p>
                    
                    <p><strong>Subject Line for Legal Notices:</strong> Please include "Legal Notice - Terms of Service" in your email subject line for prompt handling.</p>
                    
                    <p><strong>Response Time:</strong> We aim to respond to all inquiries within 48 business hours.</p>
                    
                    <p><strong>Mailing Address:</strong> For formal legal correspondence, written notices should be sent to the address provided on our Contact page.</p>
                </div>

                <div class="important-notice">
                    <h4>Acknowledgment</h4>
                    <p>BY USING THIS SITE, YOU ACKNOWLEDGE THAT YOU HAVE READ THESE TERMS OF SERVICE, UNDERSTAND THEM, AND AGREE TO BE BOUND BY THEM. IF YOU DO NOT AGREE TO THESE TERMS, YOU MUST NOT ACCESS OR USE THE SITE.</p>
                </div>

                <div style="margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <p><strong>Last Updated:</strong> January 28, 2026</p>
                </div>
            </article>
        </main>
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
            </div>
        </footer>
        <!-- Enhanced Navigation Script -->
        <script src="../static/navigation.js"></script>
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
        <style>
            .contact-method {
                background: #f8f9fa;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #6366f1;
            }

            .contact-method h3 {
                color: #333;
                margin-bottom: 1rem;
                font-size: 1.3rem;
            }

            .contact-method h4 {
                color: #555;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
                font-size: 1.1rem;
            }

            .contact-method ul {
                margin-left: 1.5rem;
                line-height: 1.8;
            }

            .contact-method ul li {
                margin-bottom: 0.5rem;
            }

            .contact-email {
                color: #6366f1;
                text-decoration: none;
                font-weight: 600;
                font-size: 1.1rem;
            }

            .contact-email:hover {
                text-decoration: underline;
            }

            .response-time {
                color: #666;
                font-style: italic;
                margin-top: 0.5rem;
                font-size: 0.9rem;
            }

            .faq-section {
                background: #f0f4ff;
                border-left-color: #8b5cf6;
            }

            .faq-item {
                margin-bottom: 1.5rem;
            }

            .faq-item:last-child {
                margin-bottom: 0;
            }

            .note {
                color: #666;
                font-size: 0.9rem;
                margin-top: 0.5rem;
            }

            .contact-footer {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 8px;
                margin-top: 2rem;
            }

            .contact-footer p {
                margin-bottom: 0.5rem;
            }

            .contact-footer p:last-child {
                margin-bottom: 0;
            }
        </style>
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
                    <a href="../terms-of-service/">Terms of Service</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>Contact Us</h2>
                <p>Get in touch with the AI Tech Blog team</p>
            </div>
            <article class="page-content">
                <p>We'd love to hear from you! Whether you have questions, feedback, or collaboration opportunities, feel free to reach out to our team.</p>
                
                <div class="contact-info">
                    <h2>Get in Touch</h2>
                    
                    <div class="contact-method">
                        <h3>📧 Email</h3>
                        <p><a href="mailto:aiblogauto@gmail.com" class="contact-email">aiblogauto@gmail.com</a></p>
                        <p class="response-time">We typically respond within 5 working days</p>
                    </div>
                    
                    <div class="contact-method">
                        <h3>💬 What We Cover</h3>
                        <ul>
                            <li><strong>General Inquiries:</strong> Questions about our content, blog topics, or AI technology</li>
                            <li><strong>Collaboration:</strong> Partnership opportunities, guest posting, or content contributions</li>
                            <li><strong>Technical Support:</strong> Issues accessing content or technical questions</li>
                            <li><strong>Feedback:</strong> Suggestions for improvement or topic requests</li>
                            <li><strong>Media & Press:</strong> Interview requests or media inquiries</li>
                        </ul>
                    </div>
                    
                    <div class="contact-method">
                        <h3>🤝 Collaboration Opportunities</h3>
                        <p>We're always interested in connecting with:</p>
                        <ul>
                            <li>AI researchers and industry professionals</li>
                            <li>Technology writers and content creators</li>
                            <li>Companies working on innovative AI solutions</li>
                            <li>Academic institutions and educational organizations</li>
                        </ul>
                    </div>
                    
                    <div class="contact-method">
                        <h3>📝 Guest Posting Guidelines</h3>
                        <p>Interested in contributing to AI Tech Blog? We welcome high-quality guest posts on topics including:</p>
                        <ul>
                            <li>Artificial Intelligence and Machine Learning</li>
                            <li>Natural Language Processing</li>
                            <li>Computer Vision and Image Recognition</li>
                            <li>AI Ethics and Responsible AI</li>
                            <li>Emerging AI Technologies and Trends</li>
                        </ul>
                        <p>Please include "Guest Post Submission" in your email subject line along with a brief outline of your proposed topic.</p>
                    </div>
                    
                    <div class="contact-method">
                        <h3>⏰ Office Hours</h3>
                        <p>Our team monitors inquiries:</p>
                        <p><strong>Monday - Friday:</strong> 9:00 AM - 6:00 PM (EAT)</p>
                        <p>Weekend messages will be reviewed on the next business day.</p>
                    </div>
                    
                    <div class="contact-method">
                        <h3>🌍 Connect With Us</h3>
                        <p>Stay updated with the latest AI news and insights:</p>
                        <ul>
                            <li>Subscribe to our newsletter for weekly AI updates</li>
                            <li>Follow our latest articles and research summaries</li>
                            <li>Join our community of AI enthusiasts and professionals</li>
                        </ul>
                    </div>
                    
                    <div class="contact-method">
                        <h3>📢 Advertising & Sponsorship</h3>
                        <p>Interested in advertising or sponsorship opportunities? We offer various options to help you reach our engaged audience of AI professionals and enthusiasts. Please email us with "Advertising Inquiry" in the subject line for more information about our rates and packages.</p>
                    </div>
                    
                    <div class="contact-method faq-section">
                        <h3>❓ Frequently Asked Questions</h3>
                        <div class="faq-item">
                            <h4>How often do you publish new content?</h4>
                            <p>We publish fresh AI-related content regularly, covering the latest developments in artificial intelligence, machine learning, and related technologies.</p>
                        </div>
                        <div class="faq-item">
                            <h4>Can I republish your content?</h4>
                            <p>Please contact us for permission and guidelines regarding content republishing. We're open to partnerships with proper attribution.</p>
                        </div>
                        <div class="faq-item">
                            <h4>Do you accept sponsored content?</h4>
                            <p>Yes, we consider sponsored content that aligns with our editorial standards and provides value to our readers. Please reach out for our guidelines and pricing.</p>
                        </div>
                    </div>
                </div>
                
                <div class="contact-footer">
                    <p><strong>Before reaching out, please check our FAQ section and existing articles – your question might already be answered!</strong></p>
                    <p>We appreciate your interest in AI Tech Blog and look forward to hearing from you.</p>
                </div>
            </article>
        </main>
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
            </div>
        </footer>
        <!-- Enhanced Navigation Script -->
        <script src="../static/navigation.js"></script>
    </body>
    </html>"""
            }

            env = Environment(loader=BaseLoader())
            templates = {}
            for name, template_str in template_strings.items():
                templates[name] = env.from_string(template_str)
            return templates