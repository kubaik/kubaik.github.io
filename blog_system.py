import os
import json
import random
import yaml
import markdown as md
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader

# === Models and Utilities (replace with your imports if they exist) ===
# from models import BlogPost
# from seo_optimizer import SEOOptimizer

class BlogPost:
    def __init__(self, title, content, slug, tags, meta_description, featured_image,
                 created_at, updated_at, seo_keywords, affiliate_links, monetization_data):
        self.title = title
        self.content = content
        self.slug = slug
        self.tags = tags
        self.meta_description = meta_description
        self.featured_image = featured_image
        self.created_at = created_at
        self.updated_at = updated_at
        self.seo_keywords = seo_keywords
        self.affiliate_links = affiliate_links
        self.monetization_data = monetization_data


class SEOOptimizer:
    @staticmethod
    def generate_sitemap(posts, base_url):
        urls = [f"<url><loc>{base_url}/{p.slug}</loc></url>" for p in posts]
        return f"<?xml version='1.0'?><urlset>{''.join(urls)}</urlset>"

    @staticmethod
    def generate_robots_txt(base_url):
        return f"User-agent: *\nAllow: /\nSitemap: {base_url}/sitemap.xml"


# === Static Site Generator ===
class StaticSiteGenerator:
    def __init__(self, blog_system):
        self.blog_system = blog_system
        self.templates = self._load_templates()
        for template in self.templates.values():
            template.globals['markdown'] = self._markdown_filter

    def _markdown_filter(self, text):
        return md.markdown(text, extensions=['codehilite', 'fenced_code'])

    def _load_templates(self) -> Dict[str, Template]:
        template_strings = {
            "base": """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{% if post %}{{ post.title }} - {% endif %}{{ site_name }}</title>
<meta name="description" content="{% if post %}{{ post.meta_description }}{% else %}{{ site_description }}{% endif %}">
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header><h1><a href="/">{{ site_name }}</a></h1></header>
<main>{% block content %}{% endblock %}</main>
<footer>&copy; {{ site_name }}</footer>
</body>
</html>""",

            "post": """{% extends "base" %}
{% block content %}
<article>
<h1>{{ post.title }}</h1>
<div>{{ post.content | safe }}</div>
</article>
{% endblock %}""",

            "index": """{% extends "base" %}
{% block content %}
<h2>Latest Posts</h2>
<ul>
{% for post in posts %}
<li><a href="/{{ post.slug }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% endblock %}"""
        }

        env = Environment(loader=BaseLoader())
        env.filters['markdown'] = lambda text: md.markdown(text, extensions=['codehilite', 'fenced_code'])
        templates = {}
        for name, template_str in template_strings.items():
            templates[name] = env.from_string(template_str)
        return templates

    def _generate_post_page(self, post: BlogPost):
        post_content_html = md.markdown(post.content, extensions=['codehilite', 'fenced_code'])
        post_dict = {
            'title': post.title,
            'slug': post.slug,
            'content': post_content_html,
            'tags': post.tags,
            'meta_description': post.meta_description,
            'created_at': post.created_at,
            'seo_keywords': post.seo_keywords,
            'affiliate_links': post.affiliate_links
        }
        post_html = self.templates['post'].render(post=post_dict)
        full_html = self.templates['base'].render(
            site_name=self.blog_system.config["site_name"],
            site_description=self.blog_system.config["site_description"],
            base_url=self.blog_system.config["base_url"],
            post=post_dict,
            content=post_html
        )
        post_dir = self.blog_system.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        with open(post_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(full_html)

    def generate_site(self):
        posts = self._get_all_posts()
        self._generate_index(posts)
        for post in posts:
            self._generate_post_page(post)
        sitemap = SEOOptimizer.generate_sitemap(posts, self.blog_system.config["base_url"])
        with open(self.blog_system.output_dir / "sitemap.xml", 'w') as f:
            f.write(sitemap)
        robots = SEOOptimizer.generate_robots_txt(self.blog_system.config["base_url"])
        with open(self.blog_system.output_dir / "robots.txt", 'w') as f:
            f.write(robots)

    # Placeholder
    def _get_all_posts(self): return []
    def _generate_index(self, posts): pass


# === Blog System ===
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
            title = await self._generate_title(topic, keywords)
            content = await self._generate_content(title, topic, keywords)
            meta_description = f"Learn about {topic} in this detailed guide."
            slug = self._create_slug(title)
            affiliate_links = []
            return BlogPost(
                title=title.strip(),
                content=content.strip(),
                slug=slug,
                tags=keywords or [topic.lower().replace(' ', '-')],
                meta_description=meta_description.strip(),
                featured_image=f"/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=keywords or [topic],
                affiliate_links=affiliate_links,
                monetization_data={"ad_slots": 3, "affiliate_count": len(affiliate_links)}
            )
        except Exception as e:
            print(f"Error generating blog post: {e}")
            raise

    async def _generate_title(self, topic: str, keywords: List[str] = None) -> str:
        return f"The Future of {topic.title()}"

    async def _generate_content(self, title: str, topic: str, keywords: List[str] = None) -> str:
        return f"## Introduction\nThis is an AI-generated blog post about {topic}."

    def _create_slug(self, title: str) -> str:
        return title.lower().replace(" ", "-")

    def save_post(self, post: BlogPost):
        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(post.content)


# === Topic Picker ===
def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    topics = config.get("content_topics", [])
    if not topics:
        raise ValueError("No content_topics found in config.yaml")
    used = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            used = json.load(f)
    available = [t for t in topics if t not in used]
    if not available:
        available = topics
        used = []
    topic = random.choice(available)
    used.append(topic)
    with open(history_file, "w") as f:
        json.dump(used, f, indent=2)
    return topic


# === CLI ===
if __name__ == "__main__":
    import sys
    config_file = "config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    blog_system = BlogSystem(config)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "auto":
            topic = pick_next_topic()
            print(f"📌 Generating post for topic: {topic}")
            import asyncio
            blog_post = asyncio.run(blog_system.generate_blog_post(topic))
            blog_system.save_post(blog_post)
            print(f"✅ Post for '{topic}' generated successfully.")
        elif mode == "init":
            print("Initializing blog system...")
