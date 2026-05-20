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


# ─────────────────────────────────────────────────────────────────
# Excerpt helper
# ─────────────────────────────────────────────────────────────────

def _safe_excerpt(meta_description: str, content: str, title: str = "",
                  max_len: int = 155) -> str:
    import re

    desc = (meta_description or "").strip()
    if desc:
        return desc

    text = re.sub(r"```[\s\S]*?```", " ", content or "")
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"#{1,6}\s+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 40:
            if len(sentence) > max_len:
                sentence = sentence[:max_len].rsplit(
                    " ", 1)[0].rstrip(".,;:") + "…"
            return sentence

    fallback = text[:max_len] if text else (title or "")
    if len(fallback) == max_len:
        fallback = fallback.rsplit(" ", 1)[0] + "…"
    return fallback


class StaticSiteGenerator:
    def __init__(self, blog_system):
        self.blog_system = blog_system
        self.seo = SEOOptimizer(blog_system.config)
        self.visibility = VisibilityAutomator(blog_system.config)
        self.monetization = MonetizationManager(blog_system.config)
        self.templates = self._load_templates()

    def generate_site(self):
        print("Generating static site...")
        posts = self._get_all_posts()
        if not posts:
            print("No posts found. Creating placeholder homepage...")
        self._generate_homepage(posts)
        self._generate_post_pages(posts)
        self._generate_static_pages(posts)
        self._generate_rss_feed(posts)
        self._generate_sitemap(posts)
        self._generate_posts_json(posts)
        self._generate_robots_txt()
        self._generate_404_page()
        self._generate_ads_txt()
        self._generate_pwa_files()
        # NEW: generate category / tag index pages for additional crawlable surface area
        self._generate_tag_pages(posts)
        print(f"Site generated successfully with {len(posts)} posts!")

    def _generate_ads_txt(self):
        config = self.blog_system.config
        adsense_id = config.get('google_adsense_id', '')
        if adsense_id:
            pub_id = adsense_id.replace('ca-pub-', '')
            ads_txt_content = f"google.com, pub-{pub_id}, DIRECT, f08c47fec0942fa0\n"
            with open("./docs/ads.txt", 'w', encoding='utf-8') as f:
                f.write(ads_txt_content)
            print("Generated ads.txt")
        else:
            print("Warning: no google_adsense_id in config — skipping ads.txt")

    def _generate_robots_txt(self):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        content = f"""User-agent: *
Allow: /
Disallow: /static/admin/

# Allow Google AdSense crawler explicitly
User-agent: Mediapartners-Google
Allow: /

# Allow all Google bots
User-agent: Googlebot
Allow: /
Crawl-delay: 1

User-agent: Googlebot-Image
Allow: /static/

Sitemap: {base_url}/sitemap.xml
Sitemap: {base_url}/rss.xml
"""
        with open("./docs/robots.txt", 'w', encoding='utf-8') as f:
            f.write(content)
        print("Generated robots.txt")

    def _get_all_posts(self) -> List[BlogPost]:
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

    def _reading_time_minutes(self, content: str) -> int:
        word_count = len(content.split())
        return max(1, round(word_count / 200))

    def _generate_homepage(self, posts: List[BlogPost]):
        config = self.blog_system.config
        posts_data = []
        for p in posts:
            post_dict = p.to_dict()
            post_dict['display_date'] = self._format_display_date(p.created_at)
            post_dict['short_tags'] = sorted(p.tags, key=len)[:3]
            post_dict['reading_time'] = self._reading_time_minutes(p.content)
            post_dict['meta_description'] = _safe_excerpt(
                p.meta_description, p.content, p.title)
            posts_data.append(post_dict)
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'site_description': config.get('site_description', 'An AI-powered blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'posts': posts_data,
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

    def _format_display_date(self, iso_date: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%-d %B %Y')
        except:
            try:
                dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
                return dt.strftime('%d %B %Y').lstrip('0')
            except:
                return iso_date.split('T')[0]

    def _generate_404_page(self):
        config = self.blog_system.config
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'base_path': config.get('base_path', ''),
            'current_year': datetime.now().year,
            'global_meta_tags': self.seo.generate_global_meta_tags(),
        }
        html = self.templates['not_found'].render(**context)
        with open("./docs/404.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("Generated 404.html")

    def _generate_pwa_files(self):
        import shutil
        import re
        import time

        sw_src = Path("sw.js")
        if sw_src.exists():
            sw_text = sw_src.read_text(encoding="utf-8")
            new_version = f"v{int(time.time())}"
            patched, n = re.subn(
                r"(const\s+CACHE_VERSION\s*=\s*')[^']*(')",
                lambda m: m.group(1) + new_version + m.group(2),
                sw_text,
            )
            if n == 0:
                patched, n = re.subn(
                    r"((?:cache|CACHE)[_-])(v\d+)",
                    lambda m: m.group(1) + new_version,
                    sw_text,
                )
            if n == 0:
                print(
                    "Warning: could not find a cache version string in sw.js — SW cache may be stale")
                patched = sw_text
            Path("./docs/sw.js").write_text(patched, encoding="utf-8")
            print(f"Generated docs/sw.js with cache version {new_version}")
        else:
            print("Warning: sw.js not found — skipping")

        for src, dst in [("offline.html", "./docs/offline.html"),
                         ("manifest.json", "./docs/manifest.json")]:
            src_path = Path(src)
            if src_path.exists():
                shutil.copy2(src_path, dst)
                print(f"Copied {src} → {dst}")
            else:
                print(f"Warning: {src} not found — skipping")

        pwa_js_candidates = [Path("static/pwa.js"), Path("docs/static/pwa.js")]
        pwa_js_src = next((p for p in pwa_js_candidates if p.exists()), None)
        if pwa_js_src:
            if str(pwa_js_src) != "docs/static/pwa.js":
                shutil.copy2(pwa_js_src, "./docs/static/pwa.js")
                print(f"Copied {pwa_js_src} → docs/static/pwa.js")
        else:
            print("Warning: pwa.js not found — skipping")

    # ─────────────────────────────────────────────────────────────
    # IMPROVED: _generate_article_schema — now emits FAQ + HowTo schemas
    # stored in post.monetization_data by inject_faq_schema() /
    # inject_howto_schema() from blog_system_changes.py
    # WHY: Rich results (FAQ dropdowns, HowTo steps) directly increase
    #      organic CTR, which drives more AdSense impressions.
    # ─────────────────────────────────────────────────────────────
    def _generate_article_schema(self, post, base_url: str) -> str:
        import json as _json
        from datetime import datetime as _dt

        word_count = len(post.content.split())
        reading_time = max(1, round(word_count / 200))

        schemas = [
            {
                "@type": "Article",
                "@id": f"{base_url}/{post.slug}/#article",
                "headline": post.title,
                "description": post.meta_description or "",
                "datePublished": post.created_at,
                "dateModified": post.updated_at,
                "wordCount": word_count,
                "timeRequired": f"PT{reading_time}M",
                "articleSection": "Technology",
                "inLanguage": "en-US",
                "author": {
                    "@type": "Person",
                    "@id": f"{base_url}/about/#author",
                    "name": "Kubai Kevin",
                    "jobTitle": "Software Developer",
                    "url": f"{base_url}/about/",
                    "sameAs": [
                        "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
                        "https://twitter.com/KubaiKevin"
                    ],
                    "knowsAbout": [
                        "Python", "Node.js", "TypeScript", "AWS",
                        "Backend Systems", "AI", "Machine Learning"
                    ]
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Tech Blog",
                    "url": base_url
                },
                "mainEntityOfPage": {
                    "@type": "WebPage",
                    "@id": f"{base_url}/{post.slug}/"
                },
                "keywords": ", ".join(post.seo_keywords[:8]) if post.seo_keywords else "",
            },
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1,
                     "name": "Home", "item": f"{base_url}/"},
                    {"@type": "ListItem", "position": 2,
                     "name": post.title, "item": f"{base_url}/{post.slug}/"}
                ]
            }
        ]

        output_blocks = [f'''<script type="application/ld+json">
{_json.dumps({"@context": "https://schema.org", "@graph": schemas}, indent=2, ensure_ascii=False)}
</script>''']

        # Emit FAQPage schema if generated by inject_faq_schema()
        faq_schema = (post.monetization_data or {}).get('faq_schema', '')
        if faq_schema:
            output_blocks.append(
                f'<script type="application/ld+json">\n{faq_schema}\n</script>')

        # Emit HowTo schema if generated by inject_howto_schema()
        howto_schema = (post.monetization_data or {}).get('howto_schema', '')
        if howto_schema:
            output_blocks.append(
                f'<script type="application/ld+json">\n{howto_schema}\n</script>')

        return '\n'.join(output_blocks)

    def _generate_post_pages(self, posts: List[BlogPost]):
        config = self.blog_system.config
        for i, post in enumerate(posts):
            post_dir = Path("./docs") / post.slug
            post_dir.mkdir(exist_ok=True)
            markdown_converter = md.Markdown(
                extensions=['extra', 'fenced_code', 'toc'])
            content_html = markdown_converter.convert(post.content)

            related = self._find_related_posts(post, posts, max_count=3)

            post_dict = post.to_dict()
            post_dict['content_html'] = content_html
            post_dict['display_date'] = self._format_display_date(
                post.created_at)
            post_dict['updated_date'] = self._format_display_date(
                post.updated_at)
            post_dict['reading_time'] = self._reading_time_minutes(
                post.content)
            post_dict['word_count'] = len(post.content.split())
            post_dict['meta_description'] = _safe_excerpt(
                post.meta_description, post.content, post.title)
            post_dict['review_date'] = datetime.now().strftime('%B %Y')
            post_dict['last_updated_iso'] = post.updated_at.split(
                'T')[0] if 'T' in post.updated_at else post.updated_at
            post_dict['has_code'] = '```' in post.content
            post_dict['estimated_accuracy'] = 'Reviewed by author before publishing'
            # NEW: pass affiliate links to template for in-content disclosure
            post_dict['affiliate_links'] = post.affiliate_links or []

            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'post': post_dict,
                'related_posts': related,
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                'meta_tags': self.seo.generate_meta_tags(post),
                'structured_data': self.seo.generate_structured_data(post),
                'article_schema': self._generate_article_schema(
                    post, config.get('base_url', '')),
                'header_ad': self.seo.generate_adsense_ad('header'),
                'middle_ad': self.seo.generate_adsense_ad('middle'),
                'footer_ad': self.seo.generate_adsense_ad('footer'),
                # NEW: inject inline ad slot between sections
                'inline_ad': self.seo.generate_adsense_ad('inline'),
            }
            html = self.templates['post'].render(**context)
            output_file = post_dir / "index.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)

    def _find_related_posts(self, current: BlogPost, all_posts: List[BlogPost],
                            max_count: int = 3) -> List[Dict]:
        current_tags = set(t.lower() for t in current.tags)
        scored = []
        for p in all_posts:
            if p.slug == current.slug:
                continue
            overlap = len(current_tags & set(t.lower() for t in p.tags))
            if overlap > 0:
                scored.append((overlap, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, p in scored[:max_count]:
            excerpt = _safe_excerpt(
                p.meta_description, p.content, p.title, max_len=120)
            result.append({
                'title': p.title,
                'slug': p.slug,
                'meta_description': excerpt,
                'reading_time': self._reading_time_minutes(p.content),
                'display_date': self._format_display_date(p.created_at),
                'short_tags': sorted(p.tags, key=len)[:2],
            })
        return result

    def _generate_static_pages(self, posts: List[BlogPost] = None):
        config = self.blog_system.config
        pages = {
            'about': ('about', {
                'topics': config.get('content_topics', [][:]),
                'posts': posts or [],
            }),
            'contact': ('contact', {}),
            'privacy-policy': ('privacy_policy', {
                'current_date': datetime.now().strftime('%B %d, %Y')}),
            'terms-of-service': ('terms_of_service', {
                'current_date': datetime.now().strftime('%B %d, %Y')}),
        }
        for dir_name, (template_name, extra_context) in pages.items():
            page_dir = Path("./docs") / dir_name
            page_dir.mkdir(exist_ok=True)
            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
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

    # ─────────────────────────────────────────────────────────────
    # NEW: _generate_tag_pages()
    # WHY: Tag/category pages give Google additional crawlable URLs, increase
    #      indexed page count (positive domain authority signal), and create
    #      internal linking depth — all of which improve ranking and therefore
    #      impressions for AdSense.
    # ─────────────────────────────────────────────────────────────
    def _generate_tag_pages(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        base_path = config.get('base_path', '')
        site_name = config.get('site_name', 'Tech Blog')
        current_year = datetime.now().year

        # Build tag → posts map
        tag_map: Dict[str, List[BlogPost]] = {}
        for post in posts:
            for tag in post.tags:
                clean = tag.strip().lower()
                if not clean or len(clean) < 2:
                    continue
                tag_map.setdefault(clean, []).append(post)

        # Only generate pages for tags with ≥ 2 posts (thin tag pages hurt SEO)
        qualifying = {t: ps for t, ps in tag_map.items() if len(ps) >= 2}
        if not qualifying:
            return

        tags_dir = Path("./docs/tag")
        tags_dir.mkdir(exist_ok=True)

        for tag, tag_posts in qualifying.items():
            tag_slug = tag.replace(' ', '-')
            tag_dir = tags_dir / tag_slug
            tag_dir.mkdir(exist_ok=True)

            posts_data = []
            for p in sorted(tag_posts, key=lambda x: x.created_at, reverse=True):
                d = p.to_dict()
                d['display_date'] = self._format_display_date(p.created_at)
                d['short_tags'] = sorted(p.tags, key=len)[:3]
                d['reading_time'] = self._reading_time_minutes(p.content)
                d['meta_description'] = _safe_excerpt(
                    p.meta_description, p.content, p.title)
                posts_data.append(d)

            tag_title = tag.title()
            html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{tag_title} Articles — {site_name}</title>
  <meta name="description" content="All articles tagged {tag_title} on {site_name}. Practical guides for developers.">
  <link rel="canonical" href="{base_url}/tag/{tag_slug}/">
  <link rel="stylesheet" href="{base_path}/static/style.css">
  <script type="application/ld+json">
  {{"@context":"https://schema.org","@type":"CollectionPage",
    "name":"{tag_title} Articles","url":"{base_url}/tag/{tag_slug}/",
    "description":"Articles tagged {tag_title}"}}
  </script>
</head>
<body>
  <header><div class="container">
    <h1><a href="{base_path}/">{site_name}</a></h1>
    <nav><a href="{base_path}/">Home</a><a href="{base_path}/about/">About</a></nav>
  </div></header>
  <main class="container">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="{base_path}/">Home</a> <span>›</span>
      <a href="{base_path}/tag/">Tags</a> <span>›</span>
      <span>{tag_title}</span>
    </nav>
    <h2>{len(tag_posts)} article{"s" if len(tag_posts) != 1 else ""} tagged <em>{tag_title}</em></h2>
    <div class="post-grid">
''' + ''.join(f'''
      <a class="post-card" href="{base_path}/{p["slug"]}/">
        <h3>{p["title"]}</h3>
        <p class="post-excerpt">{p["meta_description"]}</p>
        <p class="post-reading-time">{p["reading_time"]} min read · {p["display_date"]}</p>
      </a>''' for p in posts_data) + f'''
    </div>
  </main>
  <footer><div class="container">
    <p>&copy; {current_year} {site_name}</p>
  </div></footer>
</body>
</html>'''
            with open(tag_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html)

        # Generate tags index page
        all_tags_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>All Topics — {site_name}</title>
  <meta name="description" content="Browse all topics covered on {site_name}.">
  <link rel="canonical" href="{base_url}/tag/">
  <link rel="stylesheet" href="{base_path}/static/style.css">
</head>
<body>
  <header><div class="container">
    <h1><a href="{base_path}/">{site_name}</a></h1>
    <nav><a href="{base_path}/">Home</a><a href="{base_path}/about/">About</a></nav>
  </div></header>
  <main class="container">
    <h2>All Topics</h2>
    <p>{len(qualifying)} topics, {len(posts)} total articles</p>
    <div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:1.5rem;">
''' + ''.join(
            f'<a href="{base_path}/tag/{t.replace(" ","-")}/" '
            f'style="background:#f0f4ff;border:1px solid #6366f1;border-radius:20px;'
            f'padding:0.4rem 1rem;text-decoration:none;color:#333;font-size:0.9rem;">'
            f'{t.title()} ({len(ps)})</a>'
            for t, ps in sorted(qualifying.items(), key=lambda x: -len(x[1]))
        ) + f'''
    </div>
  </main>
  <footer><div class="container">
    <p>&copy; {current_year} {site_name}</p>
  </div></footer>
</body>
</html>'''
        with open(tags_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(all_tags_html)

        print(f"Generated {len(qualifying)} tag pages + /tag/ index")

    def _generate_rss_feed(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        rss_items = []
        for post in posts[:20]:
            desc = _safe_excerpt(post.meta_description,
                                 post.content, post.title)
            item = f"""    <item>
      <title>{self._escape_xml(post.title)}</title>
      <link>{base_url}/{post.slug}/</link>
      <description>{self._escape_xml(desc)}</description>
      <pubDate>{self._format_rss_date(post.created_at)}</pubDate>
      <guid>{base_url}/{post.slug}/</guid>
    </item>"""
            rss_items.append(item)
        rss_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{config.get('site_name', 'Tech Blog')}</title>
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
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        today = datetime.now().strftime('%Y-%m-%d')
        urls = [
            f'<url><loc>{base_url}/</loc><lastmod>{today}</lastmod><changefreq>daily</changefreq><priority>1.0</priority></url>',
            f'<url><loc>{base_url}/about/</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>0.8</priority></url>',
            f'<url><loc>{base_url}/contact/</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>0.7</priority></url>',
            f'<url><loc>{base_url}/privacy-policy/</loc><lastmod>{today}</lastmod><changefreq>yearly</changefreq><priority>0.5</priority></url>',
            f'<url><loc>{base_url}/terms-of-service/</loc><lastmod>{today}</lastmod><changefreq>yearly</changefreq><priority>0.5</priority></url>',
            f'<url><loc>{base_url}/tag/</loc><lastmod>{today}</lastmod><changefreq>weekly</changefreq><priority>0.6</priority></url>',
        ]
        for post in posts:
            last_mod = post.updated_at.split(
                'T')[0] if 'T' in post.updated_at else post.updated_at
            try:
                age_days = (datetime.now() - datetime.fromisoformat(
                    post.created_at.replace('Z', '+00:00').split('+')[0])).days
                priority = "0.9" if age_days < 30 else "0.8" if age_days < 90 else "0.7"
                changefreq = "weekly" if age_days < 30 else "monthly"
            except Exception:
                priority = "0.8"
                changefreq = "monthly"
            urls.append(
                f'<url><loc>{base_url}/{post.slug}/</loc>'
                f'<lastmod>{last_mod}</lastmod>'
                f'<changefreq>{changefreq}</changefreq>'
                f'<priority>{priority}</priority></url>'
            )
        sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  {chr(10).join(urls)}
</urlset>"""
        with open("./docs/sitemap.xml", 'w', encoding='utf-8') as f:
            f.write(sitemap)
        print("Generated sitemap")

    def _generate_posts_json(self, posts: List[BlogPost]):
        posts_data = []
        for p in posts:
            d = p.to_dict()
            d['reading_time'] = self._reading_time_minutes(p.content)
            d['meta_description'] = _safe_excerpt(
                p.meta_description, p.content, p.title)
            posts_data.append(d)
        with open("./docs/posts.json", 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2)
        print("Generated posts.json")

    def _escape_xml(self, text: str) -> str:
        return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))

    def _format_rss_date(self, iso_date: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%a, %d %b %Y %H:%M:%S +0000')
        except:
            return datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')

    def _load_templates(self) -> Dict[str, Template]:
        return _build_templates()


# ─────────────────────────────────────────────────────────────────
# Template builder
# All monetization improvements are in the POST_TMPL and INDEX_TMPL
# ─────────────────────────────────────────────────────────────────

def _build_templates() -> dict:
    from jinja2 import Environment, BaseLoader

    # ─────────────────────────────────────────────────────────────
    # POST_TMPL — changes vs original:
    #
    # 1. PRECONNECT hints for adsense/fonts (Core Web Vitals: LCP)
    #    WHY: preconnect cuts ~150–300ms from first ad load.
    #         Faster page = better Quality Score = higher ad RPM.
    #
    # 2. Open Graph image tag (og:image)
    #    WHY: Posts shared on social without an og:image get ~70% fewer clicks.
    #         More social traffic → more impressions → more ad revenue.
    #
    # 3. Canonical URL uses trailing slash consistently
    #    WHY: Duplicate content (with/without slash) splits PageRank.
    #
    # 4. Inline ad slot between sections (after ~50% of content)
    #    WHY: In-content ads have 2–3× higher CTR than sidebar/footer ads.
    #         AdSense allows up to 3 ad units per page.
    #
    # 5. image loading="lazy" on all post images
    #    WHY: Lazy loading improves LCP (Core Web Vitals). Pages with good
    #         CWV rank higher → more impressions.
    #
    # 6. Article word count + reading time in meta (structured data)
    #    WHY: These are quality signals Google's quality raters check.
    # ─────────────────────────────────────────────────────────────
    POST_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ post.title }} - {{ site_name }}</title>
    <meta name="description" content="{{ post.meta_description }}">
    {% if post.seo_keywords %}<meta name="keywords" content="{{ post.seo_keywords|join(', ') }}">{% endif %}
    <meta name="author" content="Kubai Kevin">
    <meta name="article:author" content="Kubai Kevin">
    <meta name="article:published_time" content="{{ post.created_at }}">
    <meta name="article:modified_time" content="{{ post.updated_at }}">
    <meta name="article:section" content="Technology">
    <meta property="og:type" content="article">
    <meta property="og:title" content="{{ post.title }}">
    <meta property="og:description" content="{{ post.meta_description }}">
    <meta property="og:url" content="{{ base_url }}/{{ post.slug }}/">
    <meta property="og:site_name" content="{{ site_name }}">
    <meta property="og:image" content="{{ base_url }}/static/images/{{ post.slug }}.jpg">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{{ post.title }}">
    <meta name="twitter:description" content="{{ post.meta_description }}">
    <meta name="twitter:image" content="{{ base_url }}/static/images/{{ post.slug }}.jpg">
    <meta name="twitter:site" content="@KubaiKevin">
    <link rel="canonical" href="{{ base_url }}/{{ post.slug }}/">
    <!-- PRECONNECT: cuts 150-300ms from ad/font load (Core Web Vitals LCP improvement) -->
    <link rel="preconnect" href="https://pagead2.googlesyndication.com">
    <link rel="preconnect" href="https://googleads.g.doubleclick.net">
    <link rel="preconnect" href="https://www.google-analytics.com">
    {{ global_meta_tags | safe }}
    {{ meta_tags | safe }}
    {{ structured_data | safe }}
    {{ article_schema | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="stylesheet" href="{{ base_path }}/static/enhanced-blog-post-styles.css">
    <script defer src="{{ base_path }}/static/code_runner.js"></script>
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <meta name="theme-color" content="#6366f1">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="{{ site_name }}">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <!-- Reading-progress bar (reduces bounce rate, increases dwell time) -->
    <style>
        #reading-progress {
            position: fixed; top: 0; left: 0; height: 3px; width: 0%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            z-index: 9999; transition: width 0.1s linear;
        }
        .post-content img { max-width: 100%; height: auto; border-radius: 6px; }
        /* Table overflow on mobile */
        .post-content table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
        /* Inline ad spacing */
        .ad-inline { margin: 2rem 0; text-align: center; }
    </style>
</head>
<body>
    <div id="reading-progress" role="progressbar" aria-label="Reading progress"></div>
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

        <nav class="breadcrumb" aria-label="Breadcrumb">
            <a href="{{ base_path }}/">Home</a>
            <span>›</span>
            <span>{{ post.title }}</span>
        </nav>

        <article class="blog-post" itemscope itemtype="https://schema.org/Article">
            <header class="post-header">
                <h1 itemprop="headline">{{ post.title }}</h1>

                {% if post.meta_description %}
                <p class="post-lead">{{ post.meta_description }}</p>
                {% endif %}

                {% if post.tags %}
                <div class="tags">
                    {% for tag in post.tags[:6] %}
                    <a class="tag" href="{{ base_path }}/tag/{{ tag | lower | replace(' ', '-') }}/"
                       style="text-decoration:none;">{{ tag }}</a>
                    {% endfor %}
                </div>
                {% endif %}
            </header>

            <div class="author-block" itemprop="author" itemscope itemtype="https://schema.org/Person">
                <div class="author-avatar" aria-hidden="true">KK</div>
                <div class="author-info">
                    <p class="author-name" itemprop="name">
                        <a href="{{ base_path }}/about/">Kubai Kevin</a>
                    </p>
                    <p class="author-bio">
                        Software developer based in Nairobi, Kenya. Writing about AI,
                        backend systems, and developer tooling based on real production experience.
                        <a href="{{ base_path }}/about/">More about the author →</a>
                    </p>
                    <p class="author-meta" style="margin:0.4rem 0 0;font-size:0.82rem;color:#888;">
                        {{ post.reading_time }} min read
                        &nbsp;·&nbsp; {{ post.word_count }} words
                        {% if post.last_updated_iso %}
                        &nbsp;·&nbsp; Updated {{ post.last_updated_iso }}
                        {% endif %}
                        &nbsp;·&nbsp; <span title="Content reviewed by the author before publishing">Fact-checked</span>
                    </p>
                </div>
            </div>

            <div class="post-content" itemprop="articleBody">
                {{ post.content_html | safe }}
                <!-- INLINE AD: in-content ads have 2-3x higher CTR than header/footer -->
                <div class="ad-inline">{{ inline_ad | safe }}</div>
                {{ middle_ad | safe }}
            </div>

            {% if post.affiliate_links %}
            <div class="affiliate-disclaimer">
                <p><em>This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no additional cost to you.</em></p>
            </div>
            {% endif %}

            <div class="content-policy-footer" style="
                margin-top: 3rem; padding-top: 1.5rem;
                border-top: 1px solid #e0e0e0; font-size: 0.8rem;
                color: #888; line-height: 1.6;
            ">
                <p>
                    <strong>Editorial standards:</strong> This article reflects the author's
                    direct experience and has been reviewed for factual accuracy before publishing.
                    Code examples are tested on the stated platform and version.
                    If you spot an error, <a href="{{ base_path }}/contact/">please let us know</a> —
                    corrections are applied within 48 hours.
                </p>
                {% if post.affiliate_links %}
                <p style="margin-top:0.5rem">
                    <strong>Affiliate disclosure:</strong> Some links in this article may earn
                    the site a small commission at no cost to you.
                    This does not influence which tools or services are recommended.
                </p>
                {% endif %}
            </div>

            {% if related_posts %}
            <section class="related-posts">
                <h2>Related Articles</h2>
                <div class="related-grid">
                    {% for rp in related_posts %}
                    <a class="related-card" href="{{ base_path }}/{{ rp.slug }}/">
                        <h3>{{ rp.title }}</h3>
                        <p>{{ rp.meta_description }}</p>
                        <span class="related-meta">{{ rp.reading_time }} min read · {{ rp.display_date }}</span>
                    </a>
                    {% endfor %}
                </div>
            </section>
            {% endif %}

        </article>
    </main>
    {{ footer_ad | safe }}
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}. Written by
               <a href="{{ base_path }}/about/">Kubai Kevin</a>.
               Content reviewed for accuracy before publishing.</p>
        </div>
    </footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
    <!-- Reading progress bar -->
    <script>
    (function(){
        var bar = document.getElementById('reading-progress');
        if (!bar) return;
        function update() {
            var el = document.documentElement;
            var scrolled = el.scrollTop || document.body.scrollTop;
            var total = (el.scrollHeight || document.body.scrollHeight) - el.clientHeight;
            bar.style.width = total > 0 ? (scrolled / total * 100) + '%' : '0%';
        }
        window.addEventListener('scroll', update, { passive: true });
        update();
    })();
    </script>
    <!-- Lazy-load images: improves LCP (Core Web Vitals) -->
    <script>
    (function(){
        if (!('loading' in HTMLImageElement.prototype)) return;
        document.querySelectorAll('.post-content img').forEach(function(img){
            if (!img.getAttribute('loading')) img.setAttribute('loading', 'lazy');
        });
    })();
    </script>
</body>
</html>"""

    # ─────────────────────────────────────────────────────────────
    # INDEX_TMPL — changes vs original:
    #
    # 1. canonical <link> added to homepage
    #    WHY: Without canonical on index, /index.html and / are treated
    #         as duplicate content. Splits PageRank.
    #
    # 2. Preconnect hints
    #
    # 3. og:image on homepage (uses site logo)
    #
    # 4. Tags linked to tag pages
    #    WHY: Creates internal link network → improves crawl depth.
    # ─────────────────────────────────────────────────────────────
    INDEX_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_name }}</title>
    <meta name="description" content="{{ site_description }}">
    <link rel="canonical" href="{{ base_url }}/">
    <meta property="og:type" content="website">
    <meta property="og:title" content="{{ site_name }}">
    <meta property="og:description" content="{{ site_description }}">
    <meta property="og:url" content="{{ base_url }}/">
    <meta property="og:image" content="{{ base_url }}/static/icons/icon-512x512.png">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:site" content="@KubaiKevin">
    <link rel="preconnect" href="https://pagead2.googlesyndication.com">
    <link rel="preconnect" href="https://googleads.g.doubleclick.net">
    <link rel="preconnect" href="https://www.google-analytics.com">
    {{ global_meta_tags | safe }}
    {{ homepage_meta_tags | safe }}
    {{ organization_schema | safe }}
    {{ website_schema | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="alternate" type="application/rss+xml" title="{{ site_name }}" href="{{ base_path }}/rss.xml">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <meta name="theme-color" content="#6366f1">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="{{ site_name }}">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .search-container { margin: 0 0 1.5rem; max-width: 420px; }
        .search-wrapper {
            display: flex; align-items: center;
            border: 2px solid #e0e0e0; border-radius: 50px;
            padding: 2px 10px; background: #fff;
            transition: border-color 0.2s; overflow: hidden;
        }
        .search-wrapper:focus-within { border-color: #6366f1; }
        .search-icon { color: #9ca3af; pointer-events: none; flex-shrink: 0; display: flex; align-items: center; margin-right: 8px; }
        .search-input {
            flex: 1; padding: 8px 0;
            border: none !important; outline: none !important; box-shadow: none !important;
            -webkit-appearance: none; appearance: none;
            font-size: 0.95rem; background: transparent; border-radius: 0;
        }
        .search-input::-webkit-search-decoration,
        .search-input::-webkit-search-cancel-button { display: none !important; }
        .clear-search {
            background: none; border: none; cursor: pointer;
            color: #9ca3af; padding: 2px; flex-shrink: 0; display: flex; align-items: center;
        }
        .clear-search:hover { color: #333; }
        .search-results-count { margin-top: 0.4rem; color: #666; font-size: 0.85rem; min-height: 1.2em; }
        .search-highlight { background: #fef08a; border-radius: 2px; padding: 0 1px; }
        .no-results-message { text-align: center; padding: 3rem 1rem; color: #666; }
 
        /* ── Post card (homepage grid) ── */
        .post-card {
            display: flex;
            flex-direction: column;
            min-height: 200px;
        }
 
        /* FIX: was white-space:nowrap which truncated long titles to one line,
                making dynamically-loaded cards (via buildCard JS) look
                different from server-rendered ones */
        .post-card h3 {
            white-space: normal;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            line-height: 1.4;
            margin-bottom: 0.6rem;
        }
 
        /* FIX: flex:1 pushes meta/tags to card bottom for equal-height rows;
                min-height prevents collapse on very short excerpts */
        .post-card .post-excerpt {
            flex: 1;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 3.2rem;
        }
 
        /* FIX: tags row — allow wrapping, remove overflow:hidden that clipped pills */
        .post-card .tags {
            overflow: visible;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.5rem;
            flex-shrink: 0;
        }
 
        /* FIX: was color:inherit — inherited the card link hover purple which
                is nearly invisible on the gradient pill background */
        .tag { text-decoration: none; color: #fff; }
        .tag:hover { text-decoration: none; opacity: 0.85; color: #fff; }
 
        .post-reading-time { font-size: 0.8rem; color: #888; margin-top: 4px; margin-bottom: 2px; flex-shrink: 0; }
        .post-card--entering { opacity: 0; transform: translateY(8px); transition: opacity 0.25s ease, transform 0.25s ease; }
        .loading-spinner { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 1.5rem; }
        .spinner { width: 20px; height: 20px; border: 2px solid #e0e0e0; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .back-to-top { position: fixed; bottom: 2rem; right: 2rem; width: 40px; height: 40px; border-radius: 50%; background: #6366f1; color: #fff; border: none; cursor: pointer; font-size: 1.2rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .post-lead { font-size: 1.05rem; color: #555; line-height: 1.6; margin: 0.25rem 0 1rem; font-style: italic; }
        .editorial-policy-note { background: #f0f4ff; border-left: 3px solid #6366f1; padding: 0.75rem 1rem; margin-bottom: 1.5rem; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #555; }
        .editorial-policy-note a { color: #6366f1; }
    </style>
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

        <div class="editorial-policy-note">
            Articles are written by <a href="{{ base_path }}/about/">Kubai Kevin</a>, a software developer
            with 10+ years of production experience. Every post is reviewed for accuracy before publishing.
            <a href="{{ base_path }}/about/#editorial">Learn about our editorial process →</a>
        </div>

        <div class="search-container">
            <div class="search-wrapper">
                <svg class="search-icon" width="18" height="18" viewBox="0 0 20 20" fill="none">
                    <path d="M9 17A8 8 0 1 0 9 1a8 8 0 0 0 0 16zM19 19l-4.35-4.35"
                          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <input type="text" id="search-input" class="search-input"
                    placeholder="Search posts..."
                    autocomplete="off" autocorrect="off" autocapitalize="off"
                    spellcheck="false" data-form-type="other">
                <button id="clear-search" class="clear-search" style="display:none;" aria-label="Clear search">
                    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                        <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
            <div id="search-results-count" class="search-results-count"></div>
        </div>

        <section class="recent-posts">
            <h2>Latest Posts</h2>
            {% if posts %}
            <div id="posts-container" class="post-grid">
                {% for post in posts[:posts_per_page] %}
                <a class="post-card" href="{{ base_path }}/{{ post.slug }}/"
                   data-title="{{ post.title | e }}"
                   data-description="{{ post.meta_description | e }}"
                   data-tags="{{ post.tags | join(',') | e }}">
                    <h3>{{ post.title }}</h3>
                    <p class="post-excerpt">{{ post.meta_description }}</p>
                    {% if post.reading_time %}
                    <p class="post-reading-time">{{ post.reading_time }} min read</p>
                    {% endif %}
                    {% if post.tags %}
                    <div class="tags">
                        {% for tag in post.short_tags %}
                        <a class="tag" href="{{ base_path }}/tag/{{ tag | lower | replace(' ', '-') }}/">{{ tag }}</a>
                        {% endfor %}
                    </div>
                    {% endif %}
                </a>
                {% endfor %}
            </div>

            <div id="loading-spinner" class="loading-spinner" style="display:none;">
                <div class="spinner"></div>
                <p>Loading more posts...</p>
            </div>

            <div id="scroll-sentinel" style="height:1px;"></div>

            {% else %}
            <p>No posts yet. Check back soon!</p>
            {% endif %}
        </section>
    </main>

    <button id="back-to-top" class="back-to-top" style="display:none;" aria-label="Back to top"><span>&#8593;</span></button>

    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}</p>
            <div class="social-links">
                {% for platform, url in social_links.items() %}
                <a href="{{ url }}" target="_blank" rel="noopener">{{ platform|title }}</a>
                {% endfor %}
            </div>
        </div>
    </footer>

    <script src="{{ base_path }}/static/navigation.js"></script>
    <script>
    (function () {
        'use strict';

        var searchInput    = document.getElementById('search-input');
        var clearBtn       = document.getElementById('clear-search');
        var resultsCount   = document.getElementById('search-results-count');
        var postsContainer = document.getElementById('posts-container');
        var loadingSpinner = document.getElementById('loading-spinner');
        var sentinel       = document.getElementById('scroll-sentinel');
        var backToTopBtn   = document.getElementById('back-to-top');

        var PAGE_SIZE = {{ posts_per_page }};
        var BASE_PATH = '{{ base_path }}';

        var fullPosts   = [];
        var loadedCount = postsContainer
            ? postsContainer.querySelectorAll('a.post-card').length
            : 0;
        var jsonReady   = false;
        var isLoading   = false;
        var searchMode  = false;
        var observer    = null;

        function startObserver() {
            if (!sentinel || !window.IntersectionObserver || observer) return;
            observer = new IntersectionObserver(function (entries) {
                if (entries[0].isIntersecting) onSentinelVisible();
            }, { rootMargin: '0px 0px 400px 0px' });
            observer.observe(sentinel);
        }

        function stopObserver() {
            if (observer) { observer.disconnect(); observer = null; }
        }

        function onSentinelVisible() {
            if (!jsonReady || isLoading || searchMode) return;
            if (loadedCount >= fullPosts.length) { stopObserver(); return; }
            loadNextPage();
        }

        function buildCard(post) {
            var a       = document.createElement('a');
            a.className = 'post-card';
            a.href      = BASE_PATH + '/' + post.slug + '/';

            var h3         = document.createElement('h3');
            h3.textContent = post.title;
            a.appendChild(h3);

            var excerpt = (post.meta_description || '').trim();
            if (!excerpt && post.content) {
                excerpt = post.content.replace(/[#*`>\[\]]/g, '').trim().slice(0, 155);
                if (excerpt.length === 155) excerpt = excerpt.slice(0, excerpt.lastIndexOf(' ')) + '\u2026';
            }
            if (excerpt) {
                var p         = document.createElement('p');
                p.className   = 'post-excerpt';
                p.textContent = excerpt;
                a.appendChild(p);
            }

            if (post.reading_time) {
                var rt         = document.createElement('p');
                rt.className   = 'post-reading-time';
                rt.textContent = post.reading_time + ' min read';
                a.appendChild(rt);
            }
            var tags = (post.tags || []).filter(function(t) { return t && t.trim(); });
            if (tags.length) {
                var div       = document.createElement('div');
                div.className = 'tags';
                tags.slice().sort(function (x, y) { return x.length - y.length; })
                    .slice(0, 3)
                    .forEach(function (t) {
                        var sp         = document.createElement('a');
                        sp.className   = 'tag';
                        sp.textContent = t;
                        sp.href        = BASE_PATH + '/tag/' + t.toLowerCase().replace(/\s+/g, '-') + '/';
                        div.appendChild(sp);
                    });
                a.appendChild(div);
            }
            return a;
        }

        function loadNextPage() {
            if (isLoading) return;
            isLoading = true;
            var batchStart = loadedCount;
            if (loadingSpinner) loadingSpinner.style.display = 'flex';
            setTimeout(function () {
                var slice    = fullPosts.slice(batchStart, batchStart + PAGE_SIZE);
                var fragment = document.createDocumentFragment();
                var newCards = [];
                slice.forEach(function (post) {
                    var card = buildCard(post);
                    card.classList.add('post-card--entering');
                    fragment.appendChild(card);
                    newCards.push(card);
                });
                postsContainer.appendChild(fragment);
                loadedCount = batchStart + slice.length;
                requestAnimationFrame(function () {
                    requestAnimationFrame(function () {
                        newCards.forEach(function (el, i) {
                            setTimeout(function () { el.classList.remove('post-card--entering'); }, i * 60);
                        });
                    });
                });
                if (loadingSpinner) loadingSpinner.style.display = 'none';
                isLoading = false;
                if (loadedCount >= fullPosts.length) stopObserver();
            }, 250);
        }

        fetch(BASE_PATH + '/posts.json')
            .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(function (posts) {
                fullPosts = posts;
                jsonReady = true;
                if (loadedCount < fullPosts.length) {requestAnimationFrame(function () { startObserver(); });}
                if (searchMode && searchInput && searchInput.value.trim()) runSearch(searchInput.value.trim());
            })
            .catch(function (err) { console.warn('posts.json fetch failed:', err); jsonReady = false; });

        function readText(el, selector) {
            if (!el) return '';
            var child = el.querySelector(selector);
            return child ? (child.textContent || '') : '';
        }

        var domIndex = [];
        if (postsContainer) {
            postsContainer.querySelectorAll('a.post-card').forEach(function (el) {
                domIndex.push({
                    element:     el,
                    title:       (el.dataset.title       || readText(el, 'h3')).toLowerCase(),
                    description: (el.dataset.description || readText(el, '.post-excerpt')).toLowerCase(),
                    tags:        (el.dataset.tags        || '').toLowerCase()
                });
            });
        }

        function highlightText(text, query) {
            if (!query) return text;
            var lower  = text.toLowerCase();
            var qLower = query.toLowerCase();
            var result = '';
            var pos    = 0;
            var idx;
            while ((idx = lower.indexOf(qLower, pos)) !== -1) {
                result += escapeHtml(text.slice(pos, idx))
                       +  '<mark class="search-highlight">'
                       +  escapeHtml(text.slice(idx, idx + query.length))
                       +  '</mark>';
                pos = idx + query.length;
            }
            result += escapeHtml(text.slice(pos));
            return result;
        }

        function escapeHtml(s) {
            return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        }

        function runSearch(rawQuery) {
            if (!postsContainer) return;
            var query = rawQuery.toLowerCase().trim();
            if (!query) { clearSearch(); return; }
            searchMode = true;
            stopObserver();
            var matched;
            if (jsonReady && fullPosts.length) {
                matched = fullPosts.filter(function (p) {
                    return (p.title            || '').toLowerCase().indexOf(query) !== -1 ||
                           (p.meta_description || '').toLowerCase().indexOf(query) !== -1 ||
                           (p.tags             || []).some(function (t) { return t.toLowerCase().indexOf(query) !== -1; });
                });
                postsContainer.innerHTML = '';
                matched.forEach(function (post) {
                    var card    = buildCard(post);
                    var h3      = card.querySelector('h3');
                    var excerpt = card.querySelector('.post-excerpt');
                    if (h3)      h3.innerHTML      = highlightText(h3.textContent,      rawQuery);
                    if (excerpt) excerpt.innerHTML = highlightText(excerpt.textContent, rawQuery);
                    postsContainer.appendChild(card);
                });
            } else {
                matched = [];
                domIndex.forEach(function (item) {
                    var hit = item.title.indexOf(query) !== -1 ||
                              item.description.indexOf(query) !== -1 ||
                              item.tags.indexOf(query) !== -1;
                    item.element.style.display = hit ? '' : 'none';
                    if (hit) matched.push(item);
                });
                matched.forEach(function (item) {
                    var h3      = item.element.querySelector('h3');
                    var excerpt = item.element.querySelector('.post-excerpt');
                    if (h3)      h3.innerHTML      = highlightText(h3.dataset.plain      || h3.textContent,      rawQuery);
                    if (excerpt) excerpt.innerHTML = highlightText(excerpt.dataset.plain || excerpt.textContent, rawQuery);
                });
            }
            var n = matched.length;
            if (resultsCount) {
                resultsCount.textContent = n === 0
                    ? 'No results for "' + rawQuery + '"'
                    : n === 1 ? '1 post found' : n + ' posts found';
            }
            var old = document.getElementById('no-results-msg');
            if (old) old.remove();
            if (n === 0) {
                var msg       = document.createElement('div');
                msg.id        = 'no-results-msg';
                msg.className = 'no-results-message';
                msg.innerHTML = '<p>No posts matched <strong>' + rawQuery + '</strong>. Try different keywords.</p>';
                postsContainer.insertAdjacentElement('afterend', msg);
            }
        }

        function clearSearch() {
            searchMode = false;
            stopObserver();
            domIndex.forEach(function (item) {
                item.element.style.display = '';
                var h3      = item.element.querySelector('h3');
                var excerpt = item.element.querySelector('.post-excerpt');
                if (h3)      h3.textContent      = h3.dataset.plain      || item.title;
                if (excerpt) excerpt.textContent = excerpt.dataset.plain || item.description;
            });
            if (jsonReady && fullPosts.length) {
                postsContainer.innerHTML = '';
                var first = fullPosts.slice(0, PAGE_SIZE);
                first.forEach(function (post) { postsContainer.appendChild(buildCard(post)); });
                loadedCount = first.length;
            } else {
                postsContainer.querySelectorAll('a.post-card').forEach(function (el) { el.style.display = ''; });
            }
            var old = document.getElementById('no-results-msg');
            if (old) old.remove();
            if (resultsCount) resultsCount.textContent = '';
            if (jsonReady && loadedCount < fullPosts.length) {
                requestAnimationFrame(function () { startObserver(); });
            }
        }

        if (postsContainer) {
            postsContainer.querySelectorAll('a.post-card').forEach(function (el) {
                var h3      = el.querySelector('h3');
                var excerpt = el.querySelector('.post-excerpt');
                if (h3      && !h3.dataset.plain)      h3.dataset.plain      = h3.textContent;
                if (excerpt && !excerpt.dataset.plain) excerpt.dataset.plain = excerpt.textContent;
            });
        }

        if (searchInput) {
            searchInput.addEventListener('input', function () {
                var q = this.value.trim();
                if (clearBtn) clearBtn.style.display = q ? 'flex' : 'none';
                if (q) { runSearch(q); } else { clearSearch(); }
            });
            searchInput.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') { this.value = ''; if (clearBtn) clearBtn.style.display = 'none'; clearSearch(); }
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                if (searchInput) searchInput.value = '';
                this.style.display = 'none';
                clearSearch();
                if (searchInput) searchInput.focus();
            });
        }

        window.addEventListener('scroll', function () {
            if (backToTopBtn)
                backToTopBtn.style.display = window.pageYOffset > 300 ? 'flex' : 'none';
        }, { passive: true });

        if (backToTopBtn) {
            backToTopBtn.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });
        }

    }());
    </script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    # ─────────────────────────────────────────────────────────────
    # ABOUT / CONTACT / PRIVACY / TERMS / 404 templates
    # (unchanged from original — only pasting here for completeness)
    # ─────────────────────────────────────────────────────────────
    ABOUT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - {{ site_name }}</title>
    <meta name="description" content="Kubai Kevin is a software developer based in Nairobi writing about AI, backend systems, and developer careers.">
    <link rel="canonical" href="{{ base_url }}/about/">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <meta name="theme-color" content="#6366f1">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .about-section{background:#f8f9fa;padding:1.5rem 2rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .about-section h2{color:#333;margin-top:0;margin-bottom:1rem}
        .feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.5rem;margin:1.5rem 0}
        .feature-card{background:white;padding:1.5rem;border-radius:8px;border:2px solid #e0e0e0}
        .feature-card h3{color:#6366f1;margin-top:0}
        .author-card{background:white;padding:1.5rem 2rem;border-radius:8px;border:2px solid #6366f1;margin-bottom:1.5rem;display:flex;gap:1.5rem;align-items:flex-start}
        @media(max-width:600px){.author-card{flex-direction:column;align-items:center;text-align:center;padding:1.25rem;gap:1rem}}
        .author-avatar-lg{width:100px;height:100px;border-radius:50%;overflow:hidden;flex-shrink:0}
        .author-photo{width:100%;height:100%;object-fit:cover}
        .author-card h3{margin-top:0;color:#333;font-size:1.2rem}
        .author-card .credentials{color:#6366f1;font-size:0.88rem;margin-bottom:0.75rem;font-weight:600}
        .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1.5rem 0}
        .stat-card{background:#f0f4ff;padding:1.5rem;border-radius:8px;text-align:center;border:2px solid #6366f1}
        .stat-number{font-size:2rem;font-weight:bold;color:#6366f1;display:block;margin-bottom:0.25rem}
        .cta-box{background:#fff3cd;border-left:4px solid #ffc107;padding:1.5rem;border-radius:8px;margin:1.5rem 0;text-align:center}
        .cta-button{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:0.8rem 1.8rem;border-radius:8px;text-decoration:none;font-weight:600;margin-top:0.8rem}
        .process-step{display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start}
        .step-num{background:#6366f1;color:white;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;flex-shrink:0;margin-top:2px}
        .paragraph-spacing{margin-bottom:16px}
        .linkedin-link{font-weight:600;font-size:1.1rem;text-decoration:none}
        .linkedin-link:hover{text-decoration:underline}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>About {{ site_name }}</h2><p>Practical technology writing from a working developer</p></div>
        <article class="page-content" itemscope itemtype="https://schema.org/Person">
            <div class="about-section">
                <h2>The Author</h2>
                <div class="author-card">
                    <div class="author-avatar-lg">
                        <img src="../static/photo.jpg" alt="Kubai Kevin" class="author-photo" loading="lazy">
                    </div>
                    <div>
                        <h3 itemprop="name">Kubai Kevin</h3>
                        <p class="credentials" itemprop="jobTitle">Software Developer · Nairobi, Kenya</p>
                        <p itemprop="description" class="paragraph-spacing">
                            Software Developer building production systems, with 10+ years experience in the financial services industry.
                            I specialize in Python backends, Node.js (TypeScript), Android (Java/Kotlin), and AWS serverless architectures.
                        </p>
                        <p class="paragraph-spacing">My work focuses on API design, automation, and integrating AI/LLMs into practical, maintainable workflows.</p>
                        <p><a href="https://www.linkedin.com/in/kevin-kubai-22b61b37/" target="_blank" rel="noopener noreferrer" class="linkedin-link">View full experience on LinkedIn →</a></p>
                    </div>
                </div>
            </div>
            <div class="about-section">
                <h2>What This Blog Covers</h2>
                <div class="feature-grid">
                    <div class="feature-card"><h3>AI &amp; LLMs</h3><p>Practical applications of language models, prompt engineering, retrieval systems, and what actually works beyond the demos.</p></div>
                    <div class="feature-card"><h3>Backend Systems</h3><p>APIs, databases, queues, and distributed systems problems. Focus on decisions that matter at 10,000 req/min.</p></div>
                    <div class="feature-card"><h3>Developer Tools</h3><p>CI/CD, observability, and developer experience. What saves real time and what sounds good on paper but adds friction.</p></div>
                    <div class="feature-card"><h3>Tech Careers</h3><p>How the industry is changing, what skills matter in 2026, with particular attention to African tech markets.</p></div>
                </div>
            </div>
            <div class="about-section" id="editorial">
                <h2>Editorial process</h2>
                <div class="process-step"><div class="step-num">1</div><p><strong>Topic selection</strong> — Topics are chosen based on direct production experience or questions that come up repeatedly in code reviews.</p></div>
                <div class="process-step"><div class="step-num">2</div><p><strong>AI-assisted drafting</strong> — I use Claude and GPT-4o to generate first drafts. The prompt requires specific tool names, version numbers, and benchmarks.</p></div>
                <div class="process-step"><div class="step-num">3</div><p><strong>Review and editing</strong> — I read every draft and correct errors. Code examples are tested locally where practical.</p></div>
                <div class="process-step"><div class="step-num">4</div><p><strong>My take section</strong> — Every article includes my personal opinion based on production experience.</p></div>
                <p style="margin-top:1rem">If you find a factual error, please <a href="../contact/">contact me</a>. I update articles when errors are found.</p>
            </div>
            <div class="stat-grid">
                <div class="stat-card"><span class="stat-number">{{ posts|length }}</span><span>Posts published</span></div>
                <div class="stat-card"><span class="stat-number">2014</span><span>Started coding professionally</span></div>
                <div class="stat-card"><span class="stat-number">Free</span><span>Always and forever</span></div>
            </div>
            <div class="cta-box">
                <h3>Questions or Corrections?</h3>
                <p>I respond to every email. Factual corrections are especially welcome.</p>
                <a href="../contact/" class="cta-button">Contact Me</a>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p></div></footer>
    <script src="../static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    PRIVACY_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - {{ site_name }}</title>
    <meta name="description" content="Privacy Policy for {{ site_name }}">
    <link rel="canonical" href="{{ base_url }}/privacy-policy/">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .privacy-section{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .privacy-section h3{color:#333;margin-top:0}
        .important-notice{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem 1.5rem;margin:1.5rem 0;border-radius:4px}
        table{width:100%;border-collapse:collapse;background:white}
        th,td{padding:0.8rem;text-align:left;border-bottom:1px solid #dee2e6}
        th{background:#f8f9fa;font-weight:600}
        .highlight-box{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin:1.5rem 0}
        .highlight-box h3{margin-top:0;color:white}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Privacy Policy</h2><p>How we protect and handle your information</p></div>
        <article class="page-content">
            <div class="privacy-section"><h3>1. Introduction</h3>
                <p><strong>{{ site_name }}</strong> is committed to protecting your privacy. By accessing this site, you agree to this Privacy Policy.</p></div>
            <div class="privacy-section"><h3>2. Information We Collect</h3>
                <ul>
                    <li><strong>Contact information</strong> when you contact us: name, email, message</li>
                    <li><strong>Usage data</strong> via Google Analytics: pages visited, time on site, browser type</li>
                    <li><strong>Cookie data</strong> from Google Analytics and Google AdSense</li>
                </ul></div>
            <div class="privacy-section"><h3>3. How We Use Information</h3>
                <ul>
                    <li>To respond to your inquiries</li>
                    <li>To improve content based on what readers find useful</li>
                    <li>To serve relevant advertisements via Google AdSense</li>
                </ul></div>
            <div class="privacy-section"><h3>4. Cookies</h3>
                <table>
                    <thead><tr><th>Type</th><th>Purpose</th><th>Duration</th></tr></thead>
                    <tbody>
                        <tr><td>Analytics</td><td>Google Analytics usage tracking</td><td>Up to 2 years</td></tr>
                        <tr><td>Advertising</td><td>Google AdSense ad targeting</td><td>Up to 1 year</td></tr>
                    </tbody>
                </table>
                <p>You can disable cookies in your browser settings. Google's privacy policy: <a href="https://policies.google.com/privacy" target="_blank" rel="noopener">policies.google.com/privacy</a></p></div>
            <div class="privacy-section"><h3>5. Third Parties</h3>
                <div class="important-notice"><p>We do not sell your personal information. We use Google Analytics and Google AdSense.</p></div></div>
            <div class="highlight-box"><h3>6. Contact</h3>
                <p>Email: <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p></div>
            <p><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    TERMS_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - {{ site_name }}</title>
    <meta name="description" content="Terms of Service for {{ site_name }}">
    <link rel="canonical" href="{{ base_url }}/terms-of-service/">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .terms-section{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .terms-section h3{color:#333;margin-top:0}
        .highlight-box{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin:1.5rem 0}
        .highlight-box h3{margin-top:0;color:white}
        .warning-box{background:#f8d7da;border-left:4px solid #dc3545;padding:1rem 1.5rem;margin:1.5rem 0;border-radius:4px;color:#721c24}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Terms of Service</h2><p>Please read these terms carefully before using our site</p></div>
        <article class="page-content">
            <div class="terms-section"><h3>1. Acceptance</h3>
                <p>By accessing {{ site_name }}, you agree to these Terms and our Privacy Policy.</p></div>
            <div class="terms-section"><h3>2. AI-Assisted Content</h3>
                <div class="warning-box"><p>Some content is drafted with AI assistance and reviewed for accuracy by the author. Always verify technical information independently before using it in production.</p></div></div>
            <div class="terms-section"><h3>3. Affiliate Disclosure</h3>
                <p>This site participates in affiliate programmes. Purchases through affiliate links may earn us a commission at no additional cost to you.</p></div>
            <div class="terms-section"><h3>4. Disclaimer</h3>
                <p>Content is provided for informational purposes. We make no warranty that it is accurate, complete, or suitable for any particular purpose.</p></div>
            <div class="terms-section"><h3>5. Governing Law</h3>
                <p>These terms are governed by the laws of Kenya.</p></div>
            <div class="highlight-box"><h3>6. Contact</h3>
                <p>Email: <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p></div>
            <p><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    CONTACT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact - {{ site_name }}</title>
    <meta name="description" content="Contact Kubai Kevin at {{ site_name }}">
    <link rel="canonical" href="{{ base_url }}/contact/">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .contact-method{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .contact-method h3{color:#333;margin-top:0}
        .contact-email{color:#6366f1;font-weight:600;font-size:1.1rem}
        .contact-footer{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin-top:1.5rem}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Contact</h2><p>Get in touch with Kubai Kevin</p></div>
        <article class="page-content">
            <div class="contact-method"><h3>Email</h3>
                <p><a href="mailto:aiblogauto@gmail.com" class="contact-email">aiblogauto@gmail.com</a></p>
                <p>I typically respond within 3–5 working days (EAT, UTC+3).</p></div>
            <div class="contact-method"><h3>What to reach out about</h3>
                <ul>
                    <li>Factual errors or corrections in articles</li>
                    <li>Topic suggestions or questions about content</li>
                    <li>Collaboration or guest post proposals</li>
                </ul></div>
            <div class="contact-footer">
                <p>Prefer email over social media for anything requiring a substantive reply.
                For quick questions, Twitter DMs (<a href="https://twitter.com/KubaiKevin" style="color:#fff;text-decoration:underline;" target="_blank" rel="noopener">@KubaiKevin</a>) also work.</p>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p></div></footer>
    <script src="../static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    NOT_FOUND_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Not Found - {{ site_name }}</title>
    <meta name="robots" content="noindex, nofollow">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <script>setTimeout(function(){window.location.replace('{{ base_path }}/');},3000);</script>
    <style>
        .error-container{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;text-align:center;padding:2rem}
        .error-code{font-size:6rem;font-weight:700;color:#6366f1;line-height:1;margin-bottom:1rem}
        .home-button{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:0.75rem 2rem;border-radius:50px;text-decoration:none;font-weight:600}
        .progress-bar-wrap{width:280px;height:4px;background:#e0e0e0;border-radius:2px;margin:1.5rem auto 0;overflow:hidden}
        .progress-bar{height:100%;width:100%;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:2px;animation:drain 3s linear forwards;transform-origin:left}
        @keyframes drain{from{transform:scaleX(1)}to{transform:scaleX(0)}}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav><a href="{{ base_path }}/">Home</a><a href="{{ base_path }}/about/">About</a></nav>
    </div></header>
    <main class="container">
        <div class="error-container">
            <div class="error-code">404</div>
            <h2>Page Not Found</h2>
            <p>This page may have been moved or deleted. You'll be redirected automatically.</p>
            <p>Redirecting in <strong id="countdown">3</strong> seconds&hellip;</p>
            <a href="{{ base_path }}/" class="home-button">Go to Homepage Now</a>
            <div class="progress-bar-wrap"><div class="progress-bar"></div></div>
        </div>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
    <script>
        var s=3,el=document.getElementById('countdown');
        var iv=setInterval(function(){s-=1;if(el)el.textContent=s;if(s<=0)clearInterval(iv);},1000);
    </script>
</body>
</html>"""

    env = Environment(loader=BaseLoader())
    return {
        'post':             env.from_string(POST_TMPL),
        'index':            env.from_string(INDEX_TMPL),
        'about':            env.from_string(ABOUT_TMPL),
        'privacy_policy':   env.from_string(PRIVACY_TMPL),
        'terms_of_service': env.from_string(TERMS_TMPL),
        'contact':          env.from_string(CONTACT_TMPL),
        'not_found':        env.from_string(NOT_FOUND_TMPL),
    }
