import os
import json
import random
import yaml
import markdown as md
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader
from urllib.parse import quote
import re

class BlogPost:
    def __init__(self, title, content, slug, tags, meta_description, featured_image,
                 created_at, updated_at, seo_keywords, affiliate_links=None, monetization_data=None):
        self.title = title
        self.content = content
        self.slug = slug
        self.tags = tags or []
        self.meta_description = meta_description
        self.featured_image = featured_image
        self.created_at = created_at
        self.updated_at = updated_at
        self.seo_keywords = seo_keywords or []
        self.affiliate_links = affiliate_links or []
        self.monetization_data = monetization_data or {}

    def to_dict(self):
        return {
            'title': self.title,
            'content': self.content,
            'slug': self.slug,
            'tags': self.tags,
            'meta_description': self.meta_description,
            'featured_image': self.featured_image,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'seo_keywords': self.seo_keywords,
            'affiliate_links': self.affiliate_links,
            'monetization_data': self.monetization_data
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_markdown_file(cls, md_file_path: Path, slug: str = None) -> 'BlogPost':
        """Create a BlogPost from a markdown file when post.json is missing"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        title = "Untitled Post"
        content_without_title = content
        
        if lines and lines[0].startswith('# '):
            title = lines[0][2:].strip()
            content_without_title = '\n'.join(lines[1:]).strip()
        
        if not slug:
            slug = cls._create_slug_static(title)
        
        current_time = datetime.now().isoformat()
        
        return cls(
            title=title,
            content=content_without_title,
            slug=slug,
            tags=['recovered', 'blog'],
            meta_description=f"Blog post about {title}",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=current_time,
            updated_at=current_time,
            seo_keywords=[],
            affiliate_links=[],
            monetization_data={"ad_slots": 3, "affiliate_count": 0}
        )
    
    @staticmethod
    def _create_slug_static(title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]


class MonetizationManager:
    """Handles automated monetization features"""
    
    def __init__(self, config):
        self.config = config
        self.affiliate_programs = {
            'amazon': {
                'tag': config.get('amazon_affiliate_tag', 'your-tag-20'),
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
            
            insertion_points = self._find_insertion_points(content, suggestion["keywords"])
            
            if insertion_points:
                insert_at = random.choice(insertion_points)
                lines = enhanced_content.split('\n')
                if insert_at < len(lines):
                    lines[insert_at] += f"\n\n*Recommended: {link_html}*\n"
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
        """Find good places to insert affiliate links"""
        lines = content.split('\n')
        insertion_points = []
        
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                insertion_points.append(i)
        
        return insertion_points
    
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


class SEOOptimizer:
    """Enhanced SEO optimization with automation"""
    
    def __init__(self, config):
        self.config = config
        self.google_analytics_id = config.get('google_analytics_id')
        self.google_adsense_id = config.get('google_adsense_id')
        self.google_search_console_key = config.get('google_search_console_key')
    
    def generate_structured_data(self, post) -> str:
        """Generate JSON-LD structured data for better SEO"""
        structured_data = {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "headline": post.title,
            "description": post.meta_description,
            "author": {
                "@type": "Organization",
                "name": self.config["site_name"]
            },
            "publisher": {
                "@type": "Organization",
                "name": self.config["site_name"],
                "url": self.config["base_url"]
            },
            "datePublished": post.created_at,
            "dateModified": post.updated_at,
            "url": f"{self.config['base_url']}/{post.slug}/",
            "keywords": post.seo_keywords
        }
        
        return f'<script type="application/ld+json">\n{json.dumps(structured_data, indent=2)}\n</script>'
    
    def generate_meta_tags(self, post) -> str:
        """Generate comprehensive meta tags"""
        meta_tags = f'''
    <!-- SEO Meta Tags -->
    <meta property="og:title" content="{post.title}">
    <meta property="og:description" content="{post.meta_description}">
    <meta property="og:url" content="{self.config['base_url']}/{post.slug}/">
    <meta property="og:type" content="article">
    <meta property="og:site_name" content="{self.config['site_name']}">
    
    <!-- Twitter Cards -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{post.title}">
    <meta name="twitter:description" content="{post.meta_description}">
    
    <!-- Additional SEO -->
    <meta name="robots" content="index, follow, max-image-preview:large">
    <meta name="googlebot" content="index, follow">
    <link rel="canonical" href="{self.config['base_url']}/{post.slug}/">'''
        
        if self.google_analytics_id:
            meta_tags += f'''
    
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={self.google_analytics_id}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{self.google_analytics_id}');
    </script>'''
        
        if self.google_adsense_id:
            meta_tags += f'''
    
    <!-- Google AdSense -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={self.google_adsense_id}" crossorigin="anonymous"></script>'''
        
        return meta_tags
    
    def generate_adsense_ad(self, slot_type: str = "display") -> str:
        """Generate AdSense ad unit HTML"""
        if not self.google_adsense_id:
            return '<div class="ad-placeholder"><!-- AdSense Ad Slot --></div>'
        
        ad_html = f'''
<ins class="adsbygoogle ad-{slot_type}"
     style="display:block"
     data-ad-client="{self.google_adsense_id}"
     data-ad-slot="AUTO"
     data-ad-format="auto"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({{}});
</script>'''
        
        return ad_html

    @staticmethod
    def generate_sitemap(posts, base_url):
        urls = [f"<url><loc>{base_url}/{p.slug}/</loc><lastmod>{p.updated_at.split('T')[0]}</lastmod></url>" for p in posts]
        urls.append(f"<url><loc>{base_url}/</loc><lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod></url>")
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{''.join(urls)}
</urlset>"""

    @staticmethod
    def generate_robots_txt(base_url):
        return f"""User-agent: *
Allow: /
Disallow: /static/

Sitemap: {base_url}/sitemap.xml"""


class VisibilityAutomator:
    """Automates content distribution and visibility"""
    
    def __init__(self, config):
        self.config = config
        self.social_accounts = config.get('social_accounts', {})
    
    def submit_to_search_engines(self, sitemap_url: str):
        """Submit sitemap to search engines"""
        search_engines = [
            f"https://www.google.com/ping?sitemap={quote(sitemap_url)}",
            f"https://www.bing.com/ping?sitemap={quote(sitemap_url)}"
        ]
        
        results = []
        for engine in search_engines:
            try:
                response = requests.get(engine, timeout=10)
                results.append({
                    'engine': engine.split('//')[1].split('.')[1],
                    'status': response.status_code,
                    'success': response.status_code == 200
                })
            except Exception as e:
                results.append({
                    'engine': engine.split('//')[1].split('.')[1],
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def generate_social_posts(self, post) -> dict:
        """Generate social media posts for different platforms"""
        base_text = f"New blog post: {post.title}"
        post_url = f"{self.config['base_url']}/{post.slug}/"
        
        hashtags = ['#' + tag.replace(' ', '').replace('-', '') for tag in post.tags[:3]]
        
        social_posts = {
            'twitter': f"{base_text}\n\n{post.meta_description[:100]}...\n\n{post_url} {' '.join(hashtags[:2])}",
            'linkedin': f"{base_text}\n\n{post.meta_description}\n\nRead more: {post_url}\n\n{' '.join(hashtags)}",
            'facebook': f"{base_text}\n\n{post.meta_description}\n\n{post_url}",
            'reddit_title': post.title,
            'reddit_content': f"{post.meta_description}\n\nFull article: {post_url}"
        }
        
        return social_posts
    
    def create_rss_feed(self, posts: list) -> str:
        """Generate RSS feed for the blog"""
        rss_items = []
        for post in posts[:10]:  # Latest 10 posts
            rss_items.append(f'''
        <item>
            <title><![CDATA[{post.title}]]></title>
            <description><![CDATA[{post.meta_description}]]></description>
            <link>{self.config['base_url']}/{post.slug}/</link>
            <guid>{self.config['base_url']}/{post.slug}/</guid>
            <pubDate>{self._format_rss_date(post.created_at)}</pubDate>
        </item>''')
        
        rss_feed = f'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>{self.config['site_name']}</title>
        <description>{self.config['site_description']}</description>
        <link>{self.config['base_url']}</link>
        <lastBuildDate>{self._format_rss_date(datetime.now().isoformat())}</lastBuildDate>
        <language>en-US</language>
        {''.join(rss_items)}
    </channel>
</rss>'''
        
        return rss_feed
    
    def _format_rss_date(self, iso_date: str) -> str:
        """Convert ISO date to RSS format"""
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')
        except:
            dt = datetime.now()
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')


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
                <a href="{{ base_path }}/rss.xml">RSS</a>
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
                <a href="{{ base_path }}/rss.xml">RSS</a>
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
            <div class="post-grid">
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
                <a href="{{ base_path }}/rss.xml">RSS</a>
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
            posts=[p.to_dict() for p in posts[:10]],
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
        self._generate_about_page()
        self._generate_css()
        
        for post in posts:
            self._generate_post_page(post)
        
        # Generate RSS feed (renamed from feed.xml to rss.xml)
        rss_content = self.visibility.create_rss_feed(posts)
        with open(self.blog_system.output_dir / "rss.xml", 'w', encoding='utf-8') as f:
            f.write(rss_content)
        
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
        print(f"📁 Files created: sitemap.xml, robots.txt, rss.xml")
        print(f"🔗 Social media posts saved in each post directory")
        if self.blog_system.config.get('google_adsense_id'):
            print(f"💰 Google AdSense ads integrated")

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


class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize monetization manager
        self.monetization = MonetizationManager(config)

    def cleanup_posts(self):
        """Clean up incomplete posts and recover from markdown files"""
        print("Cleaning up posts...")
        
        if not self.output_dir.exists():
            print("No docs directory found.")
            return
        
        fixed_count = 0
        removed_count = 0
        
        for post_dir in self.output_dir.iterdir():
            if not post_dir.is_dir():
                continue
            
            post_json_path = post_dir / "post.json"
            markdown_path = post_dir / "index.md"
            
            if not post_json_path.exists() and markdown_path.exists():
                try:
                    print(f"Recovering {post_dir.name}...")
                    post = BlogPost.from_markdown_file(markdown_path, post_dir.name)
                    self.save_post(post)
                    fixed_count += 1
                    print(f"Recovered: {post.title}")
                except Exception as e:
                    print(f"Failed to recover {post_dir.name}: {e}")
            
            elif not post_json_path.exists() and not markdown_path.exists():
                print(f"Removing empty directory: {post_dir.name}")
                try:
                    post_dir.rmdir()
                    removed_count += 1
                except OSError:
                    print(f"Directory not empty: {list(post_dir.iterdir())}")
        
        print(f"Cleanup complete: {fixed_count} recovered, {removed_count} removed")

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            print("No OpenAI API key found. Using fallback content generation.")
            return self._generate_fallback_post(topic)
        
        try:
            print(f"Generating content for: {topic}")
            title = await self._generate_title(topic, keywords)
            content = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(topic, title)
            slug = self._create_slug(title)
            
            if not keywords:
                keywords = await self._generate_keywords(topic, title)
            
            post = BlogPost(
                title=title.strip(),
                content=content.strip(),
                slug=slug,
                tags=keywords[:5],
                meta_description=meta_description.strip(),
                featured_image=f"/static/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=keywords,
                affiliate_links=[],
                monetization_data={}
            )
            
            # Process for monetization
            enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
                post.content, topic
            )
            post.content = enhanced_content
            post.affiliate_links = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)
            
            return post
            
        except Exception as e:
            print(f"Error generating blog post: {e}")
            print("Falling back to sample content...")
            return self._generate_fallback_post(topic)

    def _generate_fallback_post(self, topic: str) -> BlogPost:
        """Generate a fallback post when API is unavailable"""
        title = f"Understanding {topic}: A Complete Guide"
        slug = self._create_slug(title)
        
        content = f"""## Introduction

{topic} is a crucial aspect of modern technology that every developer should understand. In this comprehensive guide, we'll explore the key concepts and best practices.

## What is {topic}?

{topic} represents an important area of technology development that has gained significant traction in recent years. Understanding its core principles is essential for building effective solutions.

## Key Benefits

- **Improved Performance**: {topic} can significantly enhance system performance
- **Better Scalability**: Implementing {topic} helps applications scale more effectively  
- **Enhanced User Experience**: Users benefit from the improvements that {topic} brings
- **Cost Effectiveness**: Proper implementation can reduce operational costs

## Best Practices

### 1. Planning and Strategy

Before implementing {topic}, it's important to have a clear strategy and understanding of your requirements.

### 2. Implementation Approach

Take a systematic approach to implementation, starting with the fundamentals and building up complexity gradually.

### 3. Testing and Optimization

Regular testing and optimization ensure that your {topic} implementation continues to perform well.

## Common Challenges

When working with {topic}, developers often encounter several common challenges:

1. **Complexity Management**: Keeping implementations simple and maintainable
2. **Performance Optimization**: Ensuring optimal performance across different scenarios
3. **Integration Issues**: Seamlessly integrating with existing systems

## Conclusion

{topic} is an essential technology for modern development. By following best practices and understanding the core concepts, you can successfully implement solutions that deliver real value.

Remember to stay updated with the latest developments in {topic} as the field continues to evolve rapidly."""

        post = BlogPost(
            title=title,
            content=content,
            slug=slug,
            tags=[topic.replace(' ', '-').lower(), 'technology', 'development', 'guide'],
            meta_description=f"A comprehensive guide to {topic} covering key concepts, benefits, and best practices for developers.",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            seo_keywords=[topic.lower(), 'guide', 'tutorial', 'best practices'],
            affiliate_links=[],
            monetization_data={}
        )
        
        # Process for monetization
        enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
            post.content, topic
        )
        post.content = enhanced_content
        post.affiliate_links = affiliate_links
        post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)
        
        return post

    async def _call_openai_api(self, messages: List[Dict], max_tokens: int = 1000):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")

    async def _generate_title(self, topic: str, keywords: List[str] = None) -> str:
        keyword_text = f" Focus on keywords: {', '.join(keywords)}" if keywords else ""
        
        messages = [
            {"role": "system", "content": "You are a skilled blog title writer. Create engaging, SEO-friendly titles."},
            {"role": "user", "content": f"Generate a compelling blog post title about '{topic}'.{keyword_text} The title should be catchy, informative, and under 60 characters."}
        ]
        
        title = await self._call_openai_api(messages, max_tokens=100)
        return title.strip().strip('"')

    async def _generate_content(self, title: str, topic: str, keywords: List[str] = None) -> str:
        keyword_text = f"\nKeywords to incorporate naturally: {', '.join(keywords)}" if keywords else ""
        
        messages = [
            {"role": "system", "content": "You are an expert technical writer. Write comprehensive, well-structured blog posts in Markdown format."},
            {"role": "user", "content": f"""Write a detailed blog post with the title: "{title}"

Topic: {topic}{keyword_text}

Requirements:
- Write in Markdown format
- Include multiple sections with proper headings (##, ###)
- Write 800-1200 words
- Include practical examples and actionable advice
- Use bullet points and numbered lists where appropriate
- Add a conclusion section
- Make it informative and engaging for readers
- Use proper Markdown formatting for code blocks, links, etc.

Do not include the main title (# {title}) as it will be added automatically."""}
        ]
        
        content = await self._call_openai_api(messages, max_tokens=2000)
        return content.strip()

    async def _generate_meta_description(self, topic: str, title: str) -> str:
        messages = [
            {"role": "system", "content": "You create SEO-optimized meta descriptions."},
            {"role": "user", "content": f"Write a compelling meta description (under 160 characters) for a blog post titled '{title}' about {topic}."}
        ]
        
        description = await self._call_openai_api(messages, max_tokens=100)
        return description.strip().strip('"')

    async def _generate_keywords(self, topic: str, title: str) -> List[str]:
        messages = [
            {"role": "system", "content": "You generate relevant SEO keywords."},
            {"role": "user", "content": f"Generate 8-10 relevant SEO keywords for a blog post titled '{title}' about {topic}. Return as a comma-separated list."}
        ]
        
        keywords_text = await self._call_openai_api(messages, max_tokens=150)
        keywords = [k.strip().strip('"') for k in keywords_text.split(',')]
        return [k for k in keywords if k][:10]

    def _create_slug(self, title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]

    def save_post(self, post: BlogPost):
        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        
        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post.to_dict(), f, indent=2, ensure_ascii=False)
        
        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")
        
        print(f"Saved post: {post.title} ({post.slug})")
        if post.affiliate_links:
            print(f"  • {len(post.affiliate_links)} affiliate links added")
        print(f"  • {post.monetization_data.get('ad_slots', 0)} ad slots configured")


def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    print(f"Picking topic from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please create it first.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    topics = config.get("content_topics", [])
    if not topics:
        raise ValueError("No content_topics found in config.yaml")
    
    used = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                used = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            used = []
    
    available = [t for t in topics if t not in used]
    if not available:
        print("All topics used, resetting...")
        available = topics
        used = []
    
    topic = random.choice(available)
    used.append(topic)
    
    with open(history_file, "w") as f:
        json.dump(used, f, indent=2)
    
    print(f"Selected topic: {topic}")
    return topic


def create_sample_config():
    """Create a sample config.yaml file with monetization settings"""
    config = {
        "site_name": "AI Tech Blog",
        "site_description": "Cutting-edge insights into technology, AI, and development",
        "base_url": "https://kubaik.github.io/ai-blog-system",
        "base_path": "/ai-blog-system",
        
        # Monetization settings
        "amazon_affiliate_tag": "aiblogcontent-20",  # Replace with your Amazon affiliate tag
        "google_analytics_id": "G-DST4PJYK6V",  # Add your GA4 measurement ID
        "google_adsense_id": "ca-pub-4477679588953789",  # Add your Google AdSense ID (ca-pub-xxxxxxxxxx)
        "google_search_console_key": "",  # Add your search console verification
        
        # Social media accounts (for automated posting)
        "social_accounts": {
            "twitter": "@KubaiKevin",
            "linkedin": "your-linkedin-page",
            "facebook": "your-facebook-page"
        },
        
        "content_topics": [
            "Machine Learning Algorithms",
            "Web Development Trends", 
            "Data Science Techniques",
            "Artificial Intelligence Applications",
            "Cloud Computing Platforms",
            "Cybersecurity Best Practices",
            "Mobile App Development",
            "DevOps and CI/CD",
            "Database Optimization",
            "Frontend Frameworks",
            "Backend Architecture",
            "API Design Patterns",
            "Software Testing Strategies",
            "Performance Optimization",
            "Blockchain Technology",
            "Internet of Things (IoT)",
            "Microservices Architecture",
            "Container Technologies",
            "Serverless Computing",
            "Progressive Web Apps"
        ]
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("Created sample config.yaml file with monetization settings")
    print("\n📝 Next steps:")
    print("1. Replace 'your-tag-20' with your Amazon Associates tag")
    print("2. Add your Google Analytics 4 measurement ID")
    print("3. Add your Google AdSense ID (ca-pub-xxxxxxxxxx)")
    print("4. Update social media handles")
    print("5. Consider applying for affiliate programs like:")
    print("   • ShareASale")
    print("   • Commission Junction")
    print("   • DigitalOcean referral program")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "init":
            print("Initializing enhanced blog system...")
            create_sample_config()
            os.makedirs("docs/static", exist_ok=True)
            os.makedirs("analytics", exist_ok=True)
            print("Blog system initialized with monetization features!")
            
        elif mode == "auto":
            print("Starting automated blog generation with monetization...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found. Run 'python blog_system.py init' first.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            
            try:
                topic = pick_next_topic()
                blog_post = asyncio.run(blog_system.generate_blog_post(topic))
                blog_system.save_post(blog_post)
                
                generator = StaticSiteGenerator(blog_system)
                generator.generate_site()
                
                print(f"✅ Enhanced post '{blog_post.title}' generated successfully!")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
        elif mode == "build":
            print("Building static site with monetization features...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Enhanced site rebuilt successfully!")
            
        elif mode == "cleanup":
            print("Running cleanup with monetization enhancements...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            blog_system.cleanup_posts()
            
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Cleanup and enhanced rebuild complete!")
            
        elif mode == "debug":
            print("Debug mode with monetization analysis...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            
            print(f"Output directory: {blog_system.output_dir}")
            print(f"Directory exists: {blog_system.output_dir.exists()}")
            
            if blog_system.output_dir.exists():
                items = list(blog_system.output_dir.iterdir())
                print(f"Items in directory: {len(items)}")
                for item in items:
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        post_json = item / "post.json"
                        post_md = item / "index.md"
                        social_json = item / "social_posts.json"
                        print(f"    post.json: {'Yes' if post_json.exists() else 'No'}")
                        print(f"    index.md: {'Yes' if post_md.exists() else 'No'}")
                        print(f"    social_posts.json: {'Yes' if social_json.exists() else 'No'}")
                        if post_json.exists():
                            try:
                                with open(post_json, 'r') as f:
                                    data = json.load(f)
                                print(f"    Valid post: {data.get('title', 'Unknown')}")
                                print(f"    Affiliate links: {len(data.get('affiliate_links', []))}")
                                print(f"    Ad slots: {data.get('monetization_data', {}).get('ad_slots', 0)}")
                            except Exception as e:
                                print(f"    Invalid JSON: {e}")
            
            print("\nRunning automatic cleanup with monetization...")
            blog_system.cleanup_posts()
            
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            
        elif mode == "social":
            print("Generating social media posts for existing content...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            posts = generator._get_all_posts()
            
            visibility = VisibilityAutomator(config)
            
            print(f"Generating social media posts for {len(posts)} posts...")
            for post in posts:
                social_posts = visibility.generate_social_posts(post)
                
                post_dir = blog_system.output_dir / post.slug
                social_file = post_dir / "social_posts.json"
                with open(social_file, 'w', encoding='utf-8') as f:
                    json.dump(social_posts, f, indent=2)
                
                print(f"Social posts generated for: {post.title}")
                print(f"  Twitter: {social_posts['twitter'][:50]}...")
                print(f"  LinkedIn: {social_posts['linkedin'][:50]}...")
                print(f"  Reddit: {social_posts['reddit_title']}")
                print()
            
            print("Social media posts generated for all posts!")
            
        else:
            print("Usage: python blog_system.py [init|auto|build|cleanup|debug|social]")
            print("  init    - Initialize blog system with monetization config")
            print("  auto    - Generate new post with monetization and rebuild site")
            print("  build   - Rebuild site with monetization features")
            print("  cleanup - Fix missing files and rebuild with monetization")
            print("  debug   - Debug current state with monetization analysis")
            print("  social  - Generate social media posts for existing content")
    else:
        print("Enhanced AI Blog System with Monetization & AdSense")
        print("Usage: python blog_system.py [command]")
        print("\nAvailable commands:")
        print("  init    - Initialize blog system with monetization settings")
        print("  auto    - Generate new monetized post and rebuild site")
        print("  build   - Rebuild site with all monetization features")
        print("  cleanup - Fix posts and rebuild with enhancements")
        print("  debug   - Analyze current state and rebuild")
        print("  social  - Generate social media posts for promotion")
        print("\nMonetization features included:")
        print("  • Automated affiliate link injection")
        print("  • Google AdSense integration with responsive ads")
        print("  • Strategic ad placement slots (header, middle, footer)")
        print("  • SEO optimization with structured data")
        print("  • Social media post generation")
        print("  • RSS feed for subscribers (/rss.xml)")
        print("  • Search engine submission")
        print("  • Revenue estimation and reporting")
        print("\nSetup required:")
        print("  1. Run 'init' to create config.yaml")
        print("  2. Add your Google AdSense ID (ca-pub-xxxxxxxxxx)")
        print("  3. Add your Google Analytics ID")
        print("  4. Configure affiliate program IDs")