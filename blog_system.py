import os
import json
import random
import yaml
import markdown as md
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader
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
        
        # Extract title from first line (assuming # Title format)
        lines = content.split('\n')
        title = "Untitled Post"
        content_without_title = content
        
        if lines and lines[0].startswith('# '):
            title = lines[0][2:].strip()
            content_without_title = '\n'.join(lines[1:]).strip()
        
        # Use provided slug or create from title
        if not slug:
            slug = cls._create_slug_static(title)
        
        # Generate basic metadata
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


class SEOOptimizer:
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


class StaticSiteGenerator:
    def __init__(self, blog_system):
        self.blog_system = blog_system
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
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="canonical" href="{{ base_url }}/{{ post.slug }}/">
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
            </div>
        </article>
    </main>
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
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="canonical" href="{{ base_url }}/">
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
            <div class="debug-info">
                <p><strong>Debug Info:</strong> Posts array is empty. This means either:</p>
                <ul>
                    <li>No posts have been generated yet</li>
                    <li>Posts are not being loaded properly</li>
                    <li>The docs directory structure is incorrect</li>
                </ul>
                <p>Generated at: {{ current_year }}-{{ '%02d'|format(datetime.now().month) }}-{{ '%02d'|format(datetime.now().day) }}</p>
            </div>
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
        print(f"🔍 Looking for posts in: {self.blog_system.output_dir}")
        
        if not self.blog_system.output_dir.exists():
            print(f"❌ Output directory does not exist: {self.blog_system.output_dir}")
            return posts
        
        # List all directories in docs/
        all_dirs = list(self.blog_system.output_dir.iterdir())
        print(f"📁 Found {len(all_dirs)} items in docs directory")
        
        for item in all_dirs:
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        recovered_posts = []
        
        for post_dir in self.blog_system.output_dir.iterdir():
            if post_dir.is_dir():
                post_json_path = post_dir / "post.json"
                markdown_path = post_dir / "index.md"
                
                if post_json_path.exists():
                    # Normal case: load from post.json
                    try:
                        print(f"📖 Loading post from JSON: {post_dir}")
                        with open(post_json_path, 'r', encoding='utf-8') as f:
                            post_data = json.load(f)
                        posts.append(BlogPost.from_dict(post_data))
                        print(f"✅ Successfully loaded: {post_data.get('title', 'Unknown')}")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"❌ Could not load post from {post_dir}: {e}")
                        
                elif markdown_path.exists():
                    # Recovery case: create post from markdown
                    try:
                        print(f"🔄 Recovering post from markdown: {post_dir}")
                        post = BlogPost.from_markdown_file(markdown_path, post_dir.name)
                        posts.append(post)
                        recovered_posts.append(post)
                        print(f"✅ Successfully recovered: {post.title}")
                    except Exception as e:
                        print(f"❌ Could not recover post from {post_dir}: {e}")
                else:
                    print(f"⚠️ Skipping {post_dir}: no post.json or index.md found")
        
        # Save recovered posts as proper JSON files
        if recovered_posts:
            print(f"💾 Saving {len(recovered_posts)} recovered posts...")
            for post in recovered_posts:
                self.blog_system.save_post(post)
        
        # Sort by creation date (newest first)
        posts.sort(key=lambda p: p.created_at, reverse=True)
        print(f"📊 Total posts loaded: {len(posts)} (including {len(recovered_posts)} recovered)")
        return posts

    def _generate_post_page(self, post: BlogPost):
        print(f"📄 Generating page for: {post.title}")
        
        # Convert markdown to HTML
        post_content_html = md.markdown(post.content, extensions=['codehilite', 'fenced_code', 'tables'])
        
        post_dict = post.to_dict()
        post_dict['content_html'] = post_content_html
        
        post_html = self.templates['post'].render(
            post=post_dict,
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            current_year=datetime.now().year
        )
        
        post_dir = self.blog_system.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        
        html_file = post_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(post_html)
        
        print(f"✅ Generated: {html_file}")

    def _generate_index(self, posts: List[BlogPost]):
        print(f"🏠 Generating index page with {len(posts)} posts")
        
        index_html = self.templates['index'].render(
            posts=[p.to_dict() for p in posts[:10]],  # Show latest 10 posts
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            current_year=datetime.now().year,
            datetime=datetime  # Pass datetime for debug
        )
        
        index_file = self.blog_system.output_dir / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        print(f"✅ Generated: {index_file}")
        print(f"📏 Index HTML size: {len(index_html)} characters")

    def _generate_about_page(self):
        print("ℹ️ Generating about page")
        
        about_html = self.templates['about'].render(
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            topics=self.blog_system.config.get("content_topics", []),
            current_year=datetime.now().year
        )
        
        about_dir = self.blog_system.output_dir / "about"
        about_dir.mkdir(exist_ok=True)
        
        about_file = about_dir / "index.html"
        with open(about_file, 'w', encoding='utf-8') as f:
            f.write(about_html)
        
        print(f"✅ Generated: {about_file}")

    def _generate_css(self):
        print("🎨 Generating CSS")
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

/* Debug Info */
.debug-info {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 2rem;
}

.debug-info ul {
    margin-left: 1.5rem;
    margin-top: 0.5rem;
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
"""
        static_dir = self.blog_system.output_dir / "static"
        static_dir.mkdir(exist_ok=True)
        with open(static_dir / "style.css", 'w', encoding='utf-8') as f:
            f.write(css_content)
        print(f"✅ Generated: {static_dir / 'style.css'}")

    def generate_site(self):
        print("📄 Generating static site...")
        posts = self._get_all_posts()
        print(f"📊 Found {len(posts)} posts")
        
        # Generate all pages
        self._generate_index(posts)
        self._generate_about_page()
        self._generate_css()
        
        for post in posts:
            self._generate_post_page(post)
        
        # Generate SEO files
        sitemap = SEOOptimizer.generate_sitemap(posts, self.blog_system.config["base_url"])
        with open(self.blog_system.output_dir / "sitemap.xml", 'w') as f:
            f.write(sitemap)
        
        robots = SEOOptimizer.generate_robots_txt(self.blog_system.config["base_url"])
        with open(self.blog_system.output_dir / "robots.txt", 'w') as f:
            f.write(robots)
        
        print(f"✅ Site generated successfully with {len(posts)} posts")
        
        # List final structure
        print("\n📁 Final directory structure:")
        for item in sorted(self.blog_system.output_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(self.blog_system.output_dir)
                size = item.stat().st_size
                print(f"  {rel_path} ({size} bytes)")


class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)
        self.api_key = os.getenv("OPENAI_API_KEY")

    def cleanup_posts(self):
        """Clean up incomplete posts and recover from markdown files"""
        print("🧹 Cleaning up posts...")
        
        if not self.output_dir.exists():
            print("❌ No docs directory found.")
            return
        
        fixed_count = 0
        removed_count = 0
        
        for post_dir in self.output_dir.iterdir():
            if not post_dir.is_dir():
                continue
            
            post_json_path = post_dir / "post.json"
            markdown_path = post_dir / "index.md"
            
            # Case 1: Has markdown but no JSON - recover
            if not post_json_path.exists() and markdown_path.exists():
                try:
                    print(f"🔄 Recovering {post_dir.name}...")
                    post = BlogPost.from_markdown_file(markdown_path, post_dir.name)
                    self.save_post(post)
                    fixed_count += 1
                    print(f"✅ Recovered: {post.title}")
                except Exception as e:
                    print(f"❌ Failed to recover {post_dir.name}: {e}")
            
            # Case 2: Has neither - remove empty directory
            elif not post_json_path.exists() and not markdown_path.exists():
                print(f"🗑️ Removing empty directory: {post_dir.name}")
                try:
                    post_dir.rmdir()
                    removed_count += 1
                except OSError:
                    # Directory not empty, list contents
                    print(f"⚠️ Directory not empty: {list(post_dir.iterdir())}")
        
        print(f"✅ Cleanup complete: {fixed_count} recovered, {removed_count} removed")

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            print("❌ No OpenAI API key found. Using fallback content generation.")
            return self._generate_fallback_post(topic)
        
        try:
            print(f"🤖 Generating content for: {topic}")
            title = await self._generate_title(topic, keywords)
            content = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(topic, title)
            slug = self._create_slug(title)
            
            # Generate relevant keywords if none provided
            if not keywords:
                keywords = await self._generate_keywords(topic, title)
            
            return BlogPost(
                title=title.strip(),
                content=content.strip(),
                slug=slug,
                tags=keywords[:5],  # Limit to 5 tags
                meta_description=meta_description.strip(),
                featured_image=f"/static/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=keywords,
                affiliate_links=[],
                monetization_data={"ad_slots": 3, "affiliate_count": 0}
            )
        except Exception as e:
            print(f"❌ Error generating blog post: {e}")
            print("🔄 Falling back to sample content...")
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

        return BlogPost(
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
            monetization_data={"ad_slots": 3, "affiliate_count": 0}
        )

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
        return slug[:50]  # Limit slug length

    def save_post(self, post: BlogPost):
        # Create post directory
        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        
        # Save post data as JSON for the static generator
        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save markdown for reference
        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")
        
        print(f"💾 Saved post: {post.title} ({post.slug})")
        print(f"📁 Post directory: {post_dir}")
        print(f"📄 Files created:")
        print(f"  - post.json ({(post_dir / 'post.json').stat().st_size} bytes)")
        print(f"  - index.md ({(post_dir / 'index.md').stat().st_size} bytes)")


def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    print(f"🎯 Picking topic from {config_path}")
    
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please create it first.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    topics = config.get("content_topics", [])
    if not topics:
        raise ValueError("No content_topics found in config.yaml")
    
    print(f"📋 Available topics: {len(topics)}")
    
    # Load used topics
    used = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                used = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            used = []
    
    print(f"📚 Previously used topics: {len(used)}")
    
    # Find available topics
    available = [t for t in topics if t not in used]
    if not available:
        print("🔄 All topics used, resetting...")
        available = topics
        used = []
    
    # Pick random topic
    topic = random.choice(available)
    used.append(topic)
    
    # Save updated history
    with open(history_file, "w") as f:
        json.dump(used, f, indent=2)
    
    print(f"✅ Selected topic: {topic}")
    return topic


def create_sample_config():
    """Create a sample config.yaml file"""
    config = {
        "site_name": "AI Tech Blog",
        "site_description": "Cutting-edge insights into technology, AI, and development",
        "base_url": "https://kubaik.github.io/ai-blog-system",
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
    
    print("✅ Created sample config.yaml file")
    print("📝 Updated base_url to your GitHub Pages URL")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "init":
            print("🚀 Initializing blog system...")
            create_sample_config()
            os.makedirs("docs/static", exist_ok=True)
            print("✅ Blog system initialized!")
            print("\nNext steps:")
            print("1. Set your OPENAI_API_KEY environment variable") 
            print("2. Run 'python blog_system.py auto' to generate your first post")
            
        elif mode == "auto":
            print("🤖 Starting automated blog generation...")
            print(f"📂 Working directory: {os.getcwd()}")
            print(f"📋 Config file exists: {os.path.exists('config.yaml')}")
            print(f"🔑 OpenAI API key available: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
            
            if not os.path.exists("config.yaml"):
                print("❌ config.yaml not found. Run 'python blog_system.py init' first.")
                sys.exit(1)
            
            # Load config
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            print(f"⚙️ Loaded config: {config['site_name']}")
            
            blog_system = BlogSystem(config)
            
            try:
                # Pick topic and generate post
                topic = pick_next_topic()
                print(f"📌 Generating post for topic: {topic}")
                
                blog_post = asyncio.run(blog_system.generate_blog_post(topic))
                blog_system.save_post(blog_post)
                
                # Generate the static site
                generator = StaticSiteGenerator(blog_system)
                generator.generate_site()
                
                print(f"✅ Post '{blog_post.title}' generated and site built successfully!")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
        elif mode == "build":
            print("🔨 Building static site from existing posts...")
            
            if not os.path.exists("config.yaml"):
                print("❌ config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Site rebuilt successfully!")
            
        elif mode == "cleanup":
            print("🧹 Running cleanup to fix missing post.json files...")
            
            if not os.path.exists("config.yaml"):
                print("❌ config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            blog_system.cleanup_posts()
            
            # Rebuild site after cleanup
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Cleanup and rebuild complete!")
            
        elif mode == "debug":
            print("🔍 Debug mode - checking current state...")
            
            if not os.path.exists("config.yaml"):
                print("❌ config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            
            print(f"📂 Output directory: {blog_system.output_dir}")
            print(f"📁 Directory exists: {blog_system.output_dir.exists()}")
            
            if blog_system.output_dir.exists():
                items = list(blog_system.output_dir.iterdir())
                print(f"📊 Items in directory: {len(items)}")
                for item in items:
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        post_json = item / "post.json"
                        post_md = item / "index.md"
                        print(f"    📄 post.json: {'✅' if post_json.exists() else '❌'}")
                        print(f"    📝 index.md: {'✅' if post_md.exists() else '❌'}")
                        if post_json.exists():
                            try:
                                with open(post_json, 'r') as f:
                                    data = json.load(f)
                                print(f"    📖 Valid post: {data.get('title', 'Unknown')}")
                            except Exception as e:
                                print(f"    ❌ Invalid JSON: {e}")
            
            # Run cleanup and rebuild
            print("\n🧹 Running automatic cleanup...")
            blog_system.cleanup_posts()
            
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            
        else:
            print("Usage: python blog_system.py [init|auto|build|cleanup|debug]")
            print("  init    - Initialize blog system and create config")
            print("  auto    - Generate new post and rebuild site")
            print("  build   - Rebuild site from existing posts")
            print("  cleanup - Fix missing post.json files and rebuild")
            print("  debug   - Debug current state, cleanup, and rebuild")
    else:
        print("Usage: python blog_system.py [init|auto|build|cleanup|debug]")