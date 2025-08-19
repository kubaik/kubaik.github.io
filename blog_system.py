import os
import json
import random
import yaml
import markdown as md
import asyncio  # Fixed: was "asyncioz"
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader

#Blog post
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
        def create_base_html(content_block, page_title_prefix=""):
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title_prefix}{{{{ site_name }}}}</title>
    <meta name="description" content="{{{{ site_description }}}}">
    <link rel="stylesheet" href="/static/style.css">
    <link rel="canonical" href="{{{{ base_url }}}}/">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="/">{{{{ site_name }}}}</a></h1>
            <nav>
                <a href="/">Home</a>
                <a href="/about/">About</a>
            </nav>
        </div>
    </header>
    <main class="container">
        {content_block}
    </main>
    <footer>
        <div class="container">
            <p>&copy; {{{{ current_year }}}} {{{{ site_name }}}}. Powered by AI.</p>
        </div>
    </footer>
</body>
</html>"""

        template_strings = {
            "post": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ post.title }} - {{ site_name }}</title>
    <meta name="description" content="{{ post.meta_description }}">
    {% if post.seo_keywords %}<meta name="keywords" content="{{ post.seo_keywords|join(', ') }}">{% endif %}
    <link rel="stylesheet" href="/static/style.css">
    <link rel="canonical" href="{{ base_url }}/{{ post.slug }}/">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="/">{{ site_name }}</a></h1>
            <nav>
                <a href="/">Home</a>
                <a href="/about/">About</a>
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
    <link rel="stylesheet" href="/static/style.css">
    <link rel="canonical" href="{{ base_url }}/">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="/">{{ site_name }}</a></h1>
            <nav>
                <a href="/">Home</a>
                <a href="/about/">About</a>
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
                    <h3><a href="/{{ post.slug }}/">{{ post.title }}</a></h3>
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
    <link rel="stylesheet" href="/static/style.css">
    <link rel="canonical" href="{{ base_url }}/about/">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="/">{{ site_name }}</a></h1>
            <nav>
                <a href="/">Home</a>
                <a href="/about/">About</a>
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
        if not self.blog_system.output_dir.exists():
            return posts
        
        for post_dir in self.blog_system.output_dir.iterdir():
            if post_dir.is_dir() and (post_dir / "post.json").exists():
                try:
                    with open(post_dir / "post.json", 'r', encoding='utf-8') as f:
                        post_data = json.load(f)
                    posts.append(BlogPost.from_dict(post_data))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load post from {post_dir}: {e}")
        
        # Sort by creation date (newest first)
        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts

    def _generate_post_page(self, post: BlogPost):
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
        with open(post_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(post_html)

    def _generate_index(self, posts: List[BlogPost]):
        index_html = self.templates['index'].render(
            posts=[p.to_dict() for p in posts[:10]],  # Show latest 10 posts
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            current_year=datetime.now().year
        )
        with open(self.blog_system.output_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(index_html)

    def _generate_about_page(self):
        about_html = self.templates['about'].render(
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            topics=self.blog_system.config.get("content_topics", []),
            current_year=datetime.now().year
        )
        about_dir = self.blog_system.output_dir / "about"
        about_dir.mkdir(exist_ok=True)
        with open(about_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(about_html)

    def _generate_css(self):
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


class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)
        self.api_key = os.getenv("OPENAI_API_KEY")

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            raise ValueError("OpenAI API key not available")
        
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
            raise

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
        import re
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


def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please create it first.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    topics = config.get("content_topics", [])
    if not topics:
        raise ValueError("No content_topics found in config.yaml")
    
    # Load used topics
    used = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                used = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            used = []
    
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
    
    return topic


def create_sample_config():
    """Create a sample config.yaml file"""
    config = {
        "site_name": "AI Tech Blog",
        "site_description": "Cutting-edge insights into technology, AI, and development",
        "base_url": "https://yourusername.github.io/your-repo-name",
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
    print("📝 Please update the base_url with your GitHub Pages URL")


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
            print("1. Edit config.yaml with your GitHub Pages URL")
            print("2. Set your OPENAI_API_KEY environment variable") 
            print("3. Run 'python blog_system.py auto' to generate your first post")
            
        elif mode == "auto":
            if not os.path.exists("config.yaml"):
                print("❌ config.yaml not found. Run 'python blog_system.py init' first.")
                sys.exit(1)
            
            # Load config
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
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
                sys.exit(1)
        
        elif mode == "build":
            if not os.path.exists("config.yaml"):
                print("❌ config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Site rebuilt successfully!")
            
        else:
            print("Usage: python blog_system.py [init|auto|build]")
            print("  init  - Initialize blog system and create config")
            print("  auto  - Generate new post and rebuild site")
            print("  build - Rebuild site from existing posts")
    else:
        print("Usage: python blog_system.py [init|auto|build]")