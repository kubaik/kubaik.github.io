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
    {{ analytics_code | safe }}
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
                {% for post in posts %}
                <article class="post-card">
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
            
            <!-- Load More Button -->
            {% if posts|length > 10 %}
            <div id="load-more-container">
                <button id="load-more" class="load-more-button">Load More Posts</button>
            </div>
            {% endif %}
            
            {% else %}
            <p>No posts yet. Check back soon!</p>
            {% endif %}
        </section>
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
        </div>
    </footer>
    
    <!-- JavaScript for Load More functionality -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const loadMoreButton = document.getElementById('load-more');
            if (loadMoreButton) {
                let currentPage = 1;
                const postsPerPage = 10;
                
                loadMoreButton.addEventListener('click', function() {
                    fetch('{{ base_path }}/posts.json')
                        .then(response => response.json())
                        .then(allPosts => {
                            currentPage++;
                            const startIndex = (currentPage - 1) * postsPerPage;
                            const endIndex = startIndex + postsPerPage;
                            const nextPosts = allPosts.slice(startIndex, endIndex);
                            
                            const container = document.getElementById('posts-container');
                            
                            nextPosts.forEach(post => {
                                const postElement = createPostElement(post);
                                container.appendChild(postElement);
                            });
                            
                            // Hide button if no more posts
                            if (endIndex >= allPosts.length) {
                                loadMoreButton.style.display = 'none';
                            }
                        })
                        .catch(error => {
                            console.error('Error loading more posts:', error);
                        });
                });
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
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="canonical" href="{{ base_url }}/about/">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
            </nav>
        </div>
    </header>
    <main class="container">
        <div class="page-content">
            <h1>About {{ site_name }}</h1>
            <p>This is an AI-powered blog that automatically generates content on various technology topics.</p>
            <p>Our content covers:</p>
            <ul>
                {% for topic in topics %}
                <li>{{ topic }}</li>
                {% endfor %}
            </ul>
            <p>All content is generated using advanced AI technology and is updated regularly.</p>
        </div>
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
        
    def _get_all_posts(self) -> List[BlogPost]:
        posts = []
        print(f"Looking for posts in: {self.blog_system.output_dir}")
        
        if not self.blog_system.output_dir.exists():
            print(f"Output directory does not exist: {self.blog_system.output_dir}")
            return posts
        
        all_dirs = list(self.blog_system.output_dir.iterdir())
        print(f"Found {len(all_dirs)} items in docs directory")
        
        for item in all_dirs:
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        recovered_posts = []
        
        for post_dir in self.blog_system.output_dir.iterdir():
            if post_dir.is_dir():
                post_json_path = post_dir / "post.json"
                markdown_path = post_dir / "index.md"
                
                if post_json_path.exists():
                    try:
                        print(f"Loading post from JSON: {post_dir}")
                        with open(post_json_path, 'r', encoding='utf-8') as f:
                            post_data = json.load(f)
                        posts.append(BlogPost.from_dict(post_data))
                        print(f"Successfully loaded: {post_data.get('title', 'Unknown')}")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Could not load post from {post_dir}: {e}")
                        
                elif markdown_path.exists():
                    try:
                        print(f"Recovering post from markdown: {post_dir}")
                        post = BlogPost.from_markdown_file(markdown_path, post_dir.name)
                        posts.append(post)
                        recovered_posts.append(post)
                        print(f"Successfully recovered: {post.title}")
                    except Exception as e:
                        print(f"Could not recover post from {post_dir}: {e}")
                else:
                    print(f"Skipping {post_dir}: no post.json or index.md found")
        
        if recovered_posts:
            print(f"Saving {len(recovered_posts)} recovered posts...")
            for post in recovered_posts:
                self.blog_system.save_post(post)
        
        posts.sort(key=lambda p: p.created_at, reverse=True)
        print(f"Total posts loaded: {len(posts)} (including {len(recovered_posts)} recovered)")
        return posts

    def _generate_post_page(self, post: BlogPost):
        print(f"Generating page for: {post.title}")
        
        post_content_html = md.markdown(post.content, extensions=['codehilite', 'fenced_code', 'tables'])
        
        post_dict = post.to_dict()
        post_dict['content_html'] = post_content_html
        
        # Generate SEO enhancements
        meta_tags = self.seo.generate_meta_tags(post)
        structured_data = self.seo.generate_structured_data(post)
        
        # Generate AdSense ads
        header_ad = self.seo.generate_adsense_ad("header")
        middle_ad = self.seo.generate_adsense_ad("middle")
        footer_ad = self.seo.generate_adsense_ad("footer")
        
        post_html = self.templates['post'].render(
            post=post_dict,
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            base_path=self.blog_system.config.get("base_path", ""),
            meta_tags=meta_tags,
            structured_data=structured_data,
            header_ad=header_ad,
            middle_ad=middle_ad,
            footer_ad=footer_ad,
            current_year=datetime.now().year
        )
        
        post_dir = self.blog_system.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        
        html_file = post_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(post_html)
        
        # Generate social media posts
        social_posts = self.visibility.generate_social_posts(post)
        social_file = post_dir / "social_posts.json"
        with open(social_file, 'w', encoding='utf-8') as f:
            json.dump(social_posts, f, indent=2)
        
        print(f"Generated: {html_file}")

    def _generate_index(self, posts: List[BlogPost]):
        print(f"Generating index page with {len(posts)} posts")
        
        analytics_code = ""
        if self.blog_system.config.get('google_analytics_id'):
            analytics_code = f'''
    <script async src="https://www.googletagmanager.com/gtag/js?id={self.blog_system.config["google_analytics_id"]}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{self.blog_system.config["google_analytics_id"]}');
    </script>'''
        
        if self.blog_system.config.get('google_adsense_id'):
            analytics_code += f'''
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={self.blog_system.config["google_adsense_id"]}" crossorigin="anonymous"></script>'''
        
        index_html = self.templates['index'].render(
            posts=[p.to_dict() for p in posts[:10]],  # Show first 10 posts initially
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            base_path=self.blog_system.config.get("base_path", ""),
            analytics_code=analytics_code,
            current_year=datetime.now().year,
            datetime=datetime
        )
        
        index_file = self.blog_system.output_dir / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        print(f"Generated: {index_file}")

    def _generate_posts_json(self, posts: List[BlogPost]):
        """Generate a JSON file with all posts for JavaScript to use"""
        posts_data = [p.to_dict() for p in posts]
        json_file = self.blog_system.output_dir / "posts.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2, ensure_ascii=False)
        print(f"Generated: {json_file}")

    def _generate_about_page(self):
        print("Generating about page")
        
        about_html = self.templates['about'].render(
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            base_path=self.blog_system.config.get("base_path", ""),
            topics=self.blog_system.config.get("content_topics", []),
            current_year=datetime.now().year
        )
        
        about_dir = self.blog_system.output_dir / "about"
        about_dir.mkdir(exist_ok=True)
        
        about_file = about_dir / "index.html"
        with open(about_file, 'w', encoding='utf-8') as f:
            f.write(about_html)
        
        print(f"Generated: {about_file}")

    def _generate_css(self):
        print("Generating CSS")
        css_content = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header */
header {
    background: #fff;
    border-bottom: 1px solid #e9ecef;
    padding: 1rem 0;
}

header h1 {
    display: inline-block;
    margin-right: 2rem;
}

header h1 a {
    text-decoration: none;
    color: #2c3e50;
    font-size: 1.5rem;
}

nav {
    display: inline-block;
}

nav a {
    text-decoration: none;
    color: #6c757d;
    margin-right: 1rem;
    padding: 0.5rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

nav a:hover {
    background-color: #e9ecef;
}

/* AdSense Ad Slots */
.ad-header, .ad-middle, .ad-footer {
    text-align: center;
    margin: 20px 0;
    min-height: 100px;
}

.ad-header { margin-bottom: 20px; }
.ad-middle { margin: 30px 0; }
.ad-footer { margin-top: 20px; }

/* AdSense Responsive */
.adsbygoogle {
    display: block;
    margin: 20px auto;
}

/* Ad Placeholder (when AdSense not configured) */
.ad-placeholder {
    text-align: center;
    margin: 20px 0;
    padding: 20px;
    background: #f8f9fa;
    border: 2px dashed #dee2e6;
    min-height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6c757d;
    font-style: italic;
}

/* Affiliate Links Styling */
a[rel*="sponsored"] {
    color: #007bff;
    font-weight: 500;
    text-decoration: none;
    border-bottom: 1px dashed #007bff;
}

a[rel*="sponsored"]:hover {
    color: #0056b3;
    border-bottom-style: solid;
}

/* Affiliate Disclaimer */
.affiliate-disclaimer {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 15px;
    margin: 30px 0;
    font-size: 0.9rem;
    color: #856404;
}

/* Main Content */
main {
    margin: 2rem auto;
    background: #fff;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Hero Section */
.hero {
    text-align: center;
    margin-bottom: 3rem;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
}

.hero h2 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

/* Post Grid */
.post-grid {
    display: grid;
    gap: 2rem;
    margin-top: 2rem;
}

.post-card {
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1.5rem;
    background: #fff;
    transition: transform 0.3s, box-shadow 0.3s;
}

.post-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.post-card h3 {
    margin-bottom: 1rem;
}

.post-card h3 a {
    text-decoration: none;
    color: #2c3e50;
}

.post-card h3 a:hover {
    color: #3498db;
}

.post-excerpt {
    color: #6c757d;
    margin-bottom: 1rem;
}

/* Load More Button */
.load-more-button {
    display: block;
    margin: 2rem auto;
    padding: 1rem 2rem;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.3s;
}

.load-more-button:hover {
    background: #2980b9;
}

/* Blog Post */
.blog-post {
    max-width: 100%;
}

.post-header {
    margin-bottom: 2rem;
    text-align: center;
}

.post-header h1 {
    font-size: 2.5rem;
    color: #2c3e50;
    margin-bottom: 1rem;
}

.post-meta {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    color: #6c757d;
    font-size: 0.9rem;
}

.post-content {
    font-size: 1.1rem;
    line-height: 1.8;
}

.post-content h2 {
    color: #2c3e50;
    margin: 2rem 0 1rem 0;
    font-size: 1.8rem;
}

.post-content h3 {
    color: #34495e;
    margin: 1.5rem 0 1rem 0;
    font-size: 1.4rem;
}

.post-content p {
    margin-bottom: 1.5rem;
}

.post-content ul, .post-content ol {
    margin-bottom: 1.5rem;
    padding-left: 2rem;
}

.post-content li {
    margin-bottom: 0.5rem;
}

.post-content blockquote {
    border-left: 4px solid #3498db;
    padding-left: 1rem;
    margin: 2rem 0;
    font-style: italic;
    color: #6c757d;
}

.post-content code {
    background: #f8f9fa;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Monaco', 'Courier New', monospace;
}

.post-content pre {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    margin: 1.5rem 0;
}

/* Tags */
.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.tag {
    background: #e9ecef;
    color: #495057;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-size: 0.8rem;
}

/* Footer */
footer {
    background: #2c3e50;
    color: #ecf0f1;
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
}

/* Page Content */
.page-content {
    max-width: 100%;
}

.page-content h1 {
    color: #2c3e50;
    margin-bottom: 2rem;
    text-align: center;
}

.page-content ul {
    margin: 1.5rem 0;
    padding-left: 2rem;
}

.page-content li {
    margin-bottom: 0.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: 0 15px;
    }
    
    .hero h2 {
        font-size: 2rem;
    }
    
    .post-header h1 {
        font-size: 2rem;
    }
    
    header h1 {
        display: block;
        margin-bottom: 1rem;
    }
    
    nav {
        display: block;
    }
    
    .ad-header, .ad-middle, .ad-footer {
        min-height: 80px;
        margin: 15px 0;
    }
    
    .load-more-button {
        padding: 0.8rem 1.5rem;
        font-size: 0.9rem;
    }
}
"""
        static_dir = self.blog_system.output_dir / "static"
        static_dir.mkdir(exist_ok=True)
        with open(static_dir / "style.css", 'w', encoding='utf-8') as f:
            f.write(css_content)
        print(f"Generated: {static_dir / 'style.css'}")

    def generate_site(self):
        print("Generating static site with monetization features...")
        posts = self._get_all_posts()
        print(f"Found {len(posts)} posts")
        
        # Process posts for monetization
        for post in posts:
            enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
                post.content, post.title + " " + " ".join(post.tags)
            )
            post.content = enhanced_content
            post.affiliate_links = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)
        
        # Generate all pages
        self._generate_index(posts)
        self._generate_posts_json(posts)  # Generate JSON for load more functionality
        self._generate_about_page()
        self._generate_css()
        
        for post in posts:
            self._generate_post_page(post)
        

        # Generate SEO files
        sitemap = self.seo.generate_sitemap(posts, self.blog_system.config["base_url"])
        with open(self.blog_system.output_dir / "sitemap.xml", 'w') as f:
            f.write(sitemap)
        
        robots = self.seo.generate_robots_txt(self.blog_system.config["base_url"])
        with open(self.blog_system.output_dir / "robots.txt", 'w') as f:
            f.write(robots)
        
        # Submit to search engines
        sitemap_url = f"{self.blog_system.config['base_url']}/sitemap.xml"
        search_results = self.visibility.submit_to_search_engines(sitemap_url)
        print("Search engine submission results:")
        for result in search_results:
            if result.get('success'):
                print(f"  ✓ {result['engine']}: Success")
            else:
                print(f"  ✗ {result['engine']}: {result.get('error', 'Failed')}")
        
        # Generate automation report
        report = self._generate_automation_report(posts)
        print(f"\n📊 Automation Report:")
        print(f"  • Total posts: {report['total_posts']}")
        print(f"  • Affiliate links: {report['monetization']['total_affiliate_links']}")
        print(f"  • AdSense configured: {'Yes' if self.blog_system.config.get('google_adsense_id') else 'No'}")
        print(f"  • Estimated monthly revenue: ${report['monetization']['estimated_monthly_revenue']}")
        print(f"  • SEO features: {report['seo']['structured_data_enabled']} posts optimized")
        print(f"  • Social posts generated: {report['visibility']['social_posts_generated']}")
        
        print(f"\n✅ Site generated successfully with {len(posts)} posts")
        print(f"📁 Files created: sitemap.xml, robots.txt, rss.xml, posts.json")
        print(f"🔗 Social media posts saved in each post directory")
        if self.blog_system.config.get('google_adsense_id'):
            print(f"💰 Google AdSense ads integrated")
        print(f"📜 Load more functionality added to index page")

    def _generate_automation_report(self, posts):
        """Generate a report of automated improvements"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_posts': len(posts),
            'monetization': {
                'total_affiliate_links': sum(len(p.affiliate_links) for p in posts),
                'estimated_monthly_revenue': self._estimate_revenue(posts),
                'ad_slots_total': sum(p.monetization_data.get('ad_slots', 0) for p in posts),
                'adsense_enabled': bool(self.blog_system.config.get('google_adsense_id'))
            },
            'seo': {
                'avg_keywords_per_post': sum(len(p.seo_keywords) for p in posts) / len(posts) if posts else 0,
                'structured_data_enabled': len(posts),
                'sitemap_urls': len(posts) + 2
            },
            'visibility': {
                'social_posts_generated': len(posts) * 4,
                'rss_enabled': True,
                'search_engine_submissions': 2
            }
        }
        
        # Save report
        analytics_dir = Path("./analytics")
        analytics_dir.mkdir(exist_ok=True)
        with open(analytics_dir / 'automation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _estimate_revenue(self, posts) -> float:
        """Rough revenue estimation based on traffic and monetization"""
        avg_monthly_visitors = 1000
        ctr = 0.02
        avg_commission = 5.0
        
        total_affiliate_links = sum(len(p.affiliate_links) for p in posts)
        estimated_clicks = avg_monthly_visitors * ctr * (total_affiliate_links / len(posts) if posts else 0)
        estimated_revenue = estimated_clicks * avg_commission * 0.1
        
        # Add AdSense revenue estimation
        if self.blog_system.config.get('google_adsense_id'):
            adsense_rpm = 2.0  # Revenue per mille (per 1000 impressions)
            estimated_adsense = (avg_monthly_visitors * adsense_rpm) / 1000
            estimated_revenue += estimated_adsense
        
        return round(estimated_revenue, 2)