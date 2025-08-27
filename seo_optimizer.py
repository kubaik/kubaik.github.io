import json
from datetime import datetime
from typing import List

class SEOOptimizer:
    """Enhanced SEO optimization with automation"""
    
    def __init__(self, config):
        self.config = config
        self.google_analytics_id = config.get('google_analytics_id')
        self.google_adsense_id = config.get('google_adsense_id')
        self.google_adsense_verification = config.get('google_adsense_verification')
        self.google_search_console_key = config.get('google_search_console_key')
    
    def generate_structured_data(self, post) -> str:
        """Generate JSON-LD structured data for better SEO"""
        base_path = self.config.get("base_path", "")
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
                "url": f"{self.config['base_url']}{base_path}"
            },
            "datePublished": post.created_at,
            "dateModified": post.updated_at,
            "url": f"{self.config['base_url']}{base_path}/{post.slug}/",
            "keywords": post.seo_keywords
        }
        
        return f'<script type="application/ld+json">\n{json.dumps(structured_data, indent=2)}\n</script>'
    
    def generate_global_meta_tags(self) -> str:
        """Generate global meta tags for all pages"""
        meta_tags = ""
        
        # Add AdSense verification if available
        if self.google_adsense_verification:
            meta_tags += f'    <meta name="google-adsense-account" content="{self.google_adsense_verification}">\n'
        
        # Add Google Search Console verification if available
        if self.google_search_console_key:
            meta_tags += f'    <meta name="google-site-verification" content="{self.google_search_console_key}">\n'
        
        # Add other global verification codes
        if self.config.get('verification_code'):
            meta_tags += f'    <meta name="verification" content="{self.config["verification_code"]}">\n'
        
        if self.google_analytics_id:
            meta_tags += f'''    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={self.google_analytics_id}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{self.google_analytics_id}');
    </script>
'''
        
        if self.google_adsense_id:
            meta_tags += f'''    <!-- Google AdSense -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={self.google_adsense_id}" crossorigin="anonymous"></script>
'''
        
        return meta_tags
    
    def generate_meta_tags(self, post) -> str:
        """Generate comprehensive meta tags for individual posts"""
        base_path = self.config.get("base_path", "")
        post_url = f"{self.config['base_url']}{base_path}/{post.slug}/"
        
        meta_tags = f'''
    <!-- SEO Meta Tags -->
    <meta property="og:title" content="{post.title}">
    <meta property="og:description" content="{post.meta_description}">
    <meta property="og:url" content="{post_url}">
    <meta property="og:type" content="article">
    <meta property="og:site_name" content="{self.config['site_name']}">

    <!-- Twitter Cards -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{post.title}">
    <meta name="twitter:description" content="{post.meta_description}">

    <!-- Additional SEO -->
    <meta name="robots" content="index, follow, max-image-preview:large">
    <meta name="googlebot" content="index, follow">
    <link rel="canonical" href="{post_url}">'''
        
        # Add verification meta tag if available
        if self.config.get('verification_code'):
            meta_tags += f'\n    <!-- Verification Meta Tag -->'
            meta_tags += f'\n    <meta name="verification" content="{self.config["verification_code"]}">'
        
        # Add AdSense verification if available
        if self.google_adsense_verification:
            meta_tags += f'\n    <!-- Google AdSense Verification -->'
            meta_tags += f'\n    <meta name="google-adsense-account" content="{self.google_adsense_verification}">'
        
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

    def generate_sitemap(self, posts) -> str:
        """Generate sitemap with base_path support"""
        base_path = self.config.get("base_path", "")
        urls = [f"<url><loc>{self.config['base_url']}{base_path}/{p.slug}/</loc><lastmod>{p.updated_at.split('T')[0]}</lastmod></url>" for p in posts]
        urls.append(f"<url><loc>{self.config['base_url']}{base_path}/</loc><lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod></url>")
        urls.append(f"<url><loc>{self.config['base_url']}{base_path}/about/</loc><lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod></url>")
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{''.join(urls)}
</urlset>"""

    def generate_robots_txt(self) -> str:
        """Generate robots.txt with base_path support"""
        base_path = self.config.get("base_path", "")
        return f"""User-agent: *
Allow: /
Disallow: /static/

Sitemap: {self.config['base_url']}{base_path}/sitemap.xml"""

    def generate_breadcrumbs(self, post=None, page_title=None) -> str:
        """Generate breadcrumb navigation with structured data"""
        base_path = self.config.get("base_path", "")
        breadcrumbs = []
        
        # Home breadcrumb
        breadcrumbs.append({
            "@type": "ListItem",
            "position": 1,
            "name": "Home",
            "item": f"{self.config['base_url']}{base_path}/"
        })
        
        if post:
            breadcrumbs.append({
                "@type": "ListItem",
                "position": 2,
                "name": post.title,
                "item": f"{self.config['base_url']}{base_path}/{post.slug}/"
            })
        elif page_title:
            breadcrumbs.append({
                "@type": "ListItem",
                "position": 2,
                "name": page_title,
                "item": f"{self.config['base_url']}{base_path}/{page_title.lower()}/"
            })
        
        structured_data = {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": breadcrumbs
        }
        
        return f'<script type="application/ld+json">\n{json.dumps(structured_data, indent=2)}\n</script>'

    def generate_homepage_meta_tags(self) -> str:
        """Generate meta tags specifically for the homepage"""
        base_path = self.config.get("base_path", "")
        homepage_url = f"{self.config['base_url']}{base_path}/"
        
        meta_tags = f'''
    <!-- Homepage SEO Meta Tags -->
    <meta property="og:title" content="{self.config['site_name']}">
    <meta property="og:description" content="{self.config['site_description']}">
    <meta property="og:url" content="{homepage_url}">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="{self.config['site_name']}">

    <!-- Twitter Cards -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{self.config['site_name']}">
    <meta name="twitter:description" content="{self.config['site_description']}">

    <!-- Additional SEO -->
    <meta name="robots" content="index, follow, max-image-preview:large">
    <meta name="googlebot" content="index, follow">
    <link rel="canonical" href="{homepage_url}">'''
        
        # Add AdSense verification if available
        if self.google_adsense_verification:
            meta_tags += f'\n    <!-- Google AdSense Verification -->'
            meta_tags += f'\n    <meta name="google-adsense-account" content="{self.google_adsense_verification}">'
        
        return meta_tags

    def generate_organization_schema(self) -> str:
        """Generate organization structured data for homepage"""
        base_path = self.config.get("base_path", "")
        social_accounts = self.config.get('social_accounts', {})
        
        social_urls = []
        if social_accounts.get('twitter'):
            social_urls.append(f"https://twitter.com/{social_accounts['twitter'].replace('@', '')}")
        if social_accounts.get('facebook'):
            social_urls.append(f"https://facebook.com/{social_accounts['facebook']}")
        if social_accounts.get('linkedin'):
            social_urls.append(f"https://linkedin.com/in/{social_accounts['linkedin']}")
        
        organization_data = {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": self.config["site_name"],
            "description": self.config["site_description"],
            "url": f"{self.config['base_url']}{base_path}/",
            "sameAs": social_urls
        }
        
        return f'<script type="application/ld+json">\n{json.dumps(organization_data, indent=2)}\n</script>'

    def generate_website_schema(self) -> str:
        """Generate website structured data"""
        base_path = self.config.get("base_path", "")
        
        website_data = {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": self.config["site_name"],
            "description": self.config["site_description"],
            "url": f"{self.config['base_url']}{base_path}/",
            "potentialAction": {
                "@type": "SearchAction",
                "target": {
                    "@type": "EntryPoint",
                    "urlTemplate": f"{self.config['base_url']}{base_path}/search?q={{search_term_string}}"
                },
                "query-input": "required name=search_term_string"
            }
        }
        
        return f'<script type="application/ld+json">\n{json.dumps(website_data, indent=2)}\n</script>'

    def get_social_media_links(self) -> dict:
        """Get formatted social media links"""
        social_accounts = self.config.get('social_accounts', {})
        links = {}
        
        if social_accounts.get('twitter'):
            links['twitter'] = f"https://twitter.com/{social_accounts['twitter'].replace('@', '')}"
        if social_accounts.get('facebook'):
            links['facebook'] = f"https://facebook.com/{social_accounts['facebook']}"
        if social_accounts.get('linkedin'):
            links['linkedin'] = f"https://linkedin.com/in/{social_accounts['linkedin']}"
        
        return links

    def generate_rss_feed(self, posts, limit=20) -> str:
        """Generate RSS feed for the blog"""
        base_path = self.config.get("base_path", "")
        
        rss_items = []
        for post in posts[:limit]:
            post_url = f"{self.config['base_url']}{base_path}/{post.slug}/"
            rss_items.append(f"""
    <item>
        <title><![CDATA[{post.title}]]></title>
        <description><![CDATA[{post.meta_description}]]></description>
        <link>{post_url}</link>
        <guid>{post_url}</guid>
        <pubDate>{datetime.fromisoformat(post.created_at.replace('Z', '+00:00')).strftime('%a, %d %b %Y %H:%M:%S %z')}</pubDate>
    </item>""")
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>{self.config['site_name']}</title>
        <description>{self.config['site_description']}</description>
        <link>{self.config['base_url']}{base_path}/</link>
        <lastBuildDate>{datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')}</lastBuildDate>
        {''.join(rss_items)}
    </channel>
</rss>"""

    def optimize_images(self, content: str) -> str:
        """Add lazy loading and SEO attributes to images"""
        import re
        
        # Pattern to match img tags
        img_pattern = r'<img([^>]+)>'
        
        def add_attributes(match):
            img_tag = match.group(0)
            attrs = match.group(1)
            
            # Add lazy loading if not present
            if 'loading=' not in attrs:
                attrs += ' loading="lazy"'
            
            # Add decoding attribute
            if 'decoding=' not in attrs:
                attrs += ' decoding="async"'
            
            return f'<img{attrs}>'
        
        return re.sub(img_pattern, add_attributes, content)

    def generate_seo_report(self, posts) -> dict:
        """Generate SEO analysis report"""
        report = {
            'total_posts': len(posts),
            'posts_with_meta_description': 0,
            'posts_with_keywords': 0,
            'average_title_length': 0,
            'average_meta_description_length': 0,
            'recommendations': []
        }
        
        total_title_length = 0
        total_meta_length = 0
        
        for post in posts:
            # Check meta descriptions
            if hasattr(post, 'meta_description') and post.meta_description:
                report['posts_with_meta_description'] += 1
                total_meta_length += len(post.meta_description)
            
            # Check keywords
            if hasattr(post, 'seo_keywords') and post.seo_keywords:
                report['posts_with_keywords'] += 1
            
            # Title length
            if hasattr(post, 'title'):
                total_title_length += len(post.title)
        
        if posts:
            report['average_title_length'] = total_title_length / len(posts)
            if report['posts_with_meta_description'] > 0:
                report['average_meta_description_length'] = total_meta_length / report['posts_with_meta_description']
        
        # Generate recommendations
        if report['posts_with_meta_description'] / len(posts) < 0.8:
            report['recommendations'].append("Add meta descriptions to more posts")
        
        if report['posts_with_keywords'] / len(posts) < 0.5:
            report['recommendations'].append("Add SEO keywords to more posts")
        
        if report['average_title_length'] > 60:
            report['recommendations'].append("Consider shorter post titles for better SEO")
        
        return report