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
            'site_name': config.get('site_name', 'AI Blog'),
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
        
        pages = {
            'about': {
                'topics': config.get('content_topics', [])[:10]
            },
            'contact': {},
            'privacy_policy': {
                'current_date': datetime.now().strftime('%B %d, %Y')
            },
            'terms_of_service': {
                'current_date': datetime.now().strftime('%B %d, %Y')
            }
        }
        
        for page_name, extra_context in pages.items():
            page_dir = Path("./docs") / page_name
            page_dir.mkdir(exist_ok=True)
            
            context = {
                'site_name': config.get('site_name', 'AI Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                **extra_context
            }
            
            html = self.templates[page_name].render(**context)
            
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
        """Load all Jinja2 templates"""
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
    {{ header_ad | safe }}
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy</a>
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
                {{ middle_ad | safe }}
            </div>
            {% if post.affiliate_links %}
            <div class="affiliate-disclaimer">
                <p><em>This post contains affiliate links.</em></p>
            </div>
            {% endif %}
        </article>
    </main>
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
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy</a>
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
            <div class="post-grid">
                {% for post in posts[:10] %}
                <article class="post-card">
                    <h3><a href="{{ base_path }}/{{ post.slug }}/">{{ post.title }}</a></h3>
                    <p>{{ post.meta_description }}</p>
                    <time>{{ post.created_at.split('T')[0] }}</time>
                </article>
                {% endfor %}
            </div>
            {% else %}
            <p>No posts yet.</p>
            {% endif %}
        </section>
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
        </div>
    </footer>
</body>
</html>""",

            "about": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>About - {{ site_name }}</title>
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
            </nav>
        </div>
    </header>
    <main class="container">
        <h1>About {{ site_name }}</h1>
        <p>An AI-powered blog delivering quality content.</p>
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
        </div>
    </footer>
</body>
</html>""",

            "contact": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Contact - {{ site_name }}</title>
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
            </nav>
        </div>
    </header>
    <main class="container">
        <h1>Contact Us</h1>
        <p>Email: <a href="mailto:kevkubai@gmail.com">kevkubai@gmail.com</a></p>
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
        </div>
    </footer>
</body>
</html>""",

            "privacy_policy": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Privacy Policy - {{ site_name }}</title>
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy</a>
            </nav>
        </div>
    </header>
    <main class="container">
        <h1>Privacy Policy</h1>
        <p>Last updated: {{ current_date }}</p>
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
        </div>
    </footer>
</body>
</html>""",

            "terms_of_service": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Terms - {{ site_name }}</title>
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/terms-of-service/">Terms</a>
            </nav>
        </div>
    </header>
    <main class="container">
        <h1>Terms of Service</h1>
        <p>Last updated: {{ current_date }}</p>
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
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