import json
from datetime import datetime
from typing import List

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