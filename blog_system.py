# Essential improvements to add to your blog_system.py

# 1. ADD MARKDOWN FILTER TO JINJA2 TEMPLATES
# Add this import at the top with other imports
import markdown as md

# 2. UPDATE THE StaticSiteGenerator.__init__ method
def __init__(self, blog_system):
    self.blog_system = blog_system
    self.templates = self._load_templates()
    # Add markdown filter to templates
    for template in self.templates.values():
        template.globals['markdown'] = self._markdown_filter

def _markdown_filter(self, text):
    """Convert markdown to HTML"""
    return md.markdown(text, extensions=['codehilite', 'fenced_code'])

# 3. FIX THE TEMPLATE INHERITANCE ISSUE
# Update the _load_templates method to use proper Jinja2 Environment
def _load_templates(self) -> Dict[str, Template]:
    """Load Jinja2 templates with proper environment"""
    from jinja2 import Environment, BaseLoader
    
    # Template strings (same as before but with proper structure)
    template_strings = {
        'base': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% if post %}{{ post.title }} - {% endif %}{{ site_name }}</title>
    <meta name="description" content="{% if post %}{{ post.meta_description }}{% else %}{{ site_description }}{% endif %}">
    {% if post and post.seo_keywords %}
    <meta name="keywords" content="{{ post.seo_keywords | join(', ') }}">
    {% endif %}
    
    <!-- SEO Meta Tags -->
    <meta property="og:title" content="{% if post %}{{ post.title }}{% else %}{{ site_name }}{% endif %}">
    <meta property="og:description" content="{% if post %}{{ post.meta_description }}{% else %}{{ site_description }}{% endif %}">
    <meta property="og:type" content="{% if post %}article{% else %}website{% endif %}">
    <meta property="og:url" content="{{ base_url }}{% if post %}/{{ post.slug }}{% endif %}">
    
    <!-- Replace YOUR-ID with actual AdSense ID when you get approved -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-YOUR-ID"
            crossorigin="anonymous"></script>
    
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <header>
        <nav>
            <h1><a href="/">{{ site_name }}</a></h1>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
            </div>
        </nav>
    </header>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        <p>&copy; 2024 {{ site_name }}. All rights reserved.</p>
        <div class="monetization-links">
            <a href="/affiliate-disclosure">Affiliate Disclosure</a>
            <a href="/privacy-policy">Privacy Policy</a>
        </div>
    </footer>
    
    <!-- Replace GA_MEASUREMENT_ID with your Google Analytics ID -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'GA_MEASUREMENT_ID');
    </script>
</body>
</html>
        ''',
        
        'post': '''
<article class="blog-post">
    <header class="post-header">
        <h1>{{ post.title }}</h1>
        <div class="post-meta">
            <time datetime="{{ post.created_at }}">{{ post.created_at[:10] }}</time>
            <div class="tags">
                {% for tag in post.tags %}
                <span class="tag">#{{ tag }}</span>
                {% endfor %}
            </div>
        </div>
    </header>
    
    <!-- Ad Slot 1 (Replace YOUR-ID and YOUR-SLOT-ID when you get AdSense) -->
    <div class="ad-slot">
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="ca-pub-YOUR-ID"
             data-ad-slot="YOUR-SLOT-ID"
             data-ad-format="auto"></ins>
        <script>(adsbygoogle = window.adsbygoogle || []).push({});</script>
    </div>
    
    <div class="post-content">
        {{ post.content | safe }}
    </div>
    
    <!-- Ad Slot 2 -->
    <div class="ad-slot">
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="ca-pub-YOUR-ID"
             data-ad-slot="YOUR-SLOT-ID"
             data-ad-format="auto"></ins>
        <script>(adsbygoogle = window.adsbygoogle || []).push({});</script>
    </div>
    
    <!-- Affiliate Section -->
    {% if post.affiliate_links %}
    <div class="affiliate-section">
        <h3>📦 Recommended Resources</h3>
        {% for link in post.affiliate_links %}
        <div class="affiliate-item">
            <h4>{{ link.name }}</h4>
            <p>{{ link.description }}</p>
            <a href="{{ link.url }}" class="affiliate-link" rel="nofollow sponsored" target="_blank">
                {{ link.anchor_text }}
            </a>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</article>

<!-- Related Posts Section -->
<section class="related-posts">
    <h3>Related Posts</h3>
    <p>More content coming soon...</p>
</section>
        ''',
        
        'index': '''
<section class="hero">
    <h1>{{ site_name }}</h1>
    <p>{{ site_description }}</p>
</section>

<!-- Ad Slot -->
<div class="ad-slot">
    <ins class="adsbygoogle"
         style="display:block"
         data-ad-client="ca-pub-YOUR-ID"
         data-ad-slot="YOUR-SLOT-ID"
         data-ad-format="auto"></ins>
    <script>(adsbygoogle = window.adsbygoogle || []).push({});</script>
</div>

<section class="latest-posts">
    <h2>Latest Posts</h2>
    <div class="posts-grid">
        {% for post in posts %}
        <article class="post-card">
            <div class="post-card-content">
                <h3><a href="/{{ post.slug }}">{{ post.title }}</a></h3>
                <p>{{ post.meta_description }}</p>
                <div class="post-meta">
                    <time>{{ post.created_at[:10] }}</time>
                    {% for tag in post.tags %}
                    <span class="tag">#{{ tag }}</span>
                    {% endfor %}
                </div>
            </div>
        </article>
        {% endfor %}
    </div>
</section>
        '''
    }
    
    # Create Jinja2 environment
    env = Environment(loader=BaseLoader())
    
    # Add markdown filter
    def markdown_filter(text):
        return md.markdown(text, extensions=['codehilite', 'fenced_code'])
    
    env.filters['markdown'] = markdown_filter
    
    templates = {}
    for name, template_str in template_strings.items():
        templates[name] = env.from_string(template_str)
    
    return templates

# 4. FIX THE POST PAGE GENERATION
def _generate_post_page(self, post: BlogPost):
    """Generate individual post HTML page with proper template rendering"""
    
    # Convert markdown to HTML
    post_content_html = md.markdown(post.content, extensions=['codehilite', 'fenced_code'])
    
    # Create a copy of the post with HTML content
    post_dict = {
        'title': post.title,
        'slug': post.slug,
        'content': post_content_html,  # HTML instead of markdown
        'tags': post.tags,
        'meta_description': post.meta_description,
        'created_at': post.created_at,
        'seo_keywords': post.seo_keywords,
        'affiliate_links': post.affiliate_links
    }
    
    # Render post content
    post_html = self.templates['post'].render(post=post_dict)
    
    # Wrap in base template
    full_html = self.templates['base'].render(
        site_name=self.blog_system.config.site_name,
        site_description=self.blog_system.config.site_description,
        base_url=self.blog_system.config.base_url,
        post=post_dict,
        content=post_html
    )
    
    # Create post directory
    post_dir = self.blog_system.output_dir / post.slug
    post_dir.mkdir(exist_ok=True)
    
    with open(post_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(full_html)

# 5. ADD SITEMAP GENERATION
def generate_site(self):
    """Generate complete static site with sitemap"""
    posts = self._get_all_posts()
    
    # Generate pages
    self._generate_index(posts)
    for post in posts:
        self._generate_post_page(post)
    self._generate_css()
    self._generate_additional_pages()
    
    # Generate sitemap
    sitemap = SEOOptimizer.generate_sitemap(posts, self.blog_system.config.base_url)
    with open(self.blog_system.output_dir / "sitemap.xml", 'w') as f:
        f.write(sitemap)
    
    # Generate robots.txt
    robots = SEOOptimizer.generate_robots_txt(self.blog_system.config.base_url)
    with open(self.blog_system.output_dir / "robots.txt", 'w') as f:
        f.write(robots)

# 6. IMPROVE ERROR HANDLING IN CONTENT GENERATION
async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
    """Generate a complete blog post with improved error handling"""
    
    if not self.api_key:
        raise ValueError("OpenAI API key not available")
    
    try:
        # Generate title
        title = await self._generate_title(topic, keywords)
        
        # Generate main content with better prompt
        content = await self._generate_content(title, topic, keywords)
        
        # Generate meta description
        meta_description = await self._generate_meta_description(title, topic)
        
        # Generate SEO keywords
        seo_keywords = await self._generate_seo_keywords(title, topic, keywords)
        
        # Create slug
        slug = self._create_slug(title)
        
        # Generate affiliate opportunities
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
    """Generate title with better prompt"""
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
    """Generate main content with better structure"""
    keywords_text = f"Keywords to include naturally: {', '.join(keywords)}" if keywords else ""
    
    prompt = f"""Write a comprehensive, engaging blog post with the title: "{title}"
Topic: {topic}
{keywords_text}

Requirements:
- 1000-1500 words (comprehensive but not too long)
- Use markdown formatting with ## for subheadings
- Include 4-6 subheadings that break up the content
- Write in conversational, engaging tone
- Include actionable tips or insights
- Add a strong introduction and conclusion
- Make it valuable and informative
- Include relevant examples where appropriate

Structure:
## Introduction
[engaging opening that hooks the reader]

## [Subheading 1]
[valuable content section]

## [Subheading 2] 
[more valuable content]

[Continue with more sections...]

## Conclusion
[summarize key points and include call-to-action]

Write the complete blog post now:"""
    
    return await self._call_openai(prompt, max_tokens=2000)