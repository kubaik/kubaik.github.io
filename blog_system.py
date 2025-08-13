import os
import markdown as md
from datetime import datetime
from typing import Dict, List
from jinja2 import Template, Environment, BaseLoader
from pathlib import Path

# Assuming BlogPost and SEOOptimizer are defined elsewhere in your project
# from models import BlogPost
# from seo_optimizer import SEOOptimizer

class StaticSiteGenerator:
    def __init__(self, blog_system):
        self.blog_system = blog_system
        self.templates = self._load_templates()
        # Add markdown filter to all templates
        for template in self.templates.values():
            template.globals['markdown'] = self._markdown_filter

    def _markdown_filter(self, text):
        """Convert markdown to HTML"""
        return md.markdown(text, extensions=['codehilite', 'fenced_code'])

    def _load_templates(self) -> Dict[str, Template]:
        """Load Jinja2 templates with proper environment"""
        template_strings = {
            'base': '''... (base HTML template from your instructions) ...''',
            'post': '''... (post HTML template from your instructions) ...''',
            'index': '''... (index HTML template from your instructions) ...'''
        }

        env = Environment(loader=BaseLoader())

        def markdown_filter(text):
            return md.markdown(text, extensions=['codehilite', 'fenced_code'])

        env.filters['markdown'] = markdown_filter

        templates = {}
        for name, template_str in template_strings.items():
            templates[name] = env.from_string(template_str)

        return templates

    def _generate_post_page(self, post):
        """Generate individual post HTML page with proper template rendering"""
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
            site_name=self.blog_system.config.site_name,
            site_description=self.blog_system.config.site_description,
            base_url=self.blog_system.config.base_url,
            post=post_dict,
            content=post_html
        )

        post_dir = self.blog_system.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        with open(post_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(full_html)

    def generate_site(self):
        """Generate complete static site with sitemap"""
        posts = self._get_all_posts()
        self._generate_index(posts)
        for post in posts:
            self._generate_post_page(post)
        self._generate_css()
        self._generate_additional_pages()

        sitemap = SEOOptimizer.generate_sitemap(posts, self.blog_system.config.base_url)
        with open(self.blog_system.output_dir / "sitemap.xml", 'w') as f:
            f.write(sitemap)

        robots = SEOOptimizer.generate_robots_txt(self.blog_system.config.base_url)
        with open(self.blog_system.output_dir / "robots.txt", 'w') as f:
            f.write(robots)

    # Placeholder for functions not shown in your snippet
    def _get_all_posts(self): ...
    def _generate_index(self, posts): ...
    def _generate_css(self): ...
    def _generate_additional_pages(self): ...


class BlogSystem:
    async def generate_blog_post(self, topic: str, keywords: List[str] = None):
        """Generate a complete blog post with improved error handling"""
        if not self.api_key:
            raise ValueError("OpenAI API key not available")

        try:
            title = await self._generate_title(topic, keywords)
            content = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(title, topic)
            seo_keywords = await self._generate_seo_keywords(title, topic, keywords)
            slug = self._create_slug(title)
            affiliate_links = await self._generate_affiliate_opportunities(topic, content)

            return BlogPost(
                title=title.strip(),
                content=content.strip(),
                slug=slug,
                tags=keywords or [topic.lower().replace(' ', '-')],
                meta_description=meta_description.strip(),
                featured_image=f"/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=seo_keywords,
                affiliate_links=affiliate_links,
                monetization_data={"ad_slots": 3, "affiliate_count": len(affiliate_links)}
            )

        except Exception as e:
            print(f"Error generating blog post: {e}")
            raise

    async def _generate_title(self, topic: str, keywords: List[str] = None) -> str:
        keywords_text = f"Keywords: {', '.join(keywords)}" if keywords else ""
        prompt = f"""Create an engaging, SEO-optimized blog post title about: {topic}
{keywords_text}

Requirements:
- Make it clickable and compelling
- 60 characters or less for SEO
- Include main keyword naturally
- Avoid clickbait, be authentic

Return only the title, nothing else."""
        return await self._call_openai(prompt, max_tokens=100)

    async def _generate_content(self, title: str, topic: str, keywords: List[str] = None) -> str:
        keywords_text = f"Keywords to include naturally: {', '.join(keywords)}" if keywords else ""
        prompt = f"""Write a comprehensive, engaging blog post with the title: "{title}"
Topic: {topic}
{keywords_text}

Requirements:
- 1000-1500 words
- Use markdown formatting
- Include 4-6 subheadings
- Engaging tone
- Actionable tips
- Intro & conclusion

Structure:
## Introduction
...

Write the complete blog post now:"""
        return await self._call_openai(prompt, max_tokens=2000)

    # Placeholder for other methods
    def _create_slug(self, title: str) -> str: ...
    async def _generate_meta_description(self, title, topic): ...
    async def _generate_seo_keywords(self, title, topic, keywords): ...
    async def _generate_affiliate_opportunities(self, topic, content): ...
    async def _call_openai(self, prompt, max_tokens): ...


if __name__ == "__main__":
    # CLI handling example
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("Initializing blog system...")
