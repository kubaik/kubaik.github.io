import json
from datetime import datetime
from typing import List

class SEOOptimizer:
    """Enhanced SEO optimization with proper AdSense verification"""
    
    def __init__(self, config):
        self.config = config
        self.google_analytics_id = config.get('google_analytics_id')
        # Fix: Use single AdSense ID for both ads and verification
        self.google_adsense_id = config.get('google_adsense_id')
        # Google Search Console verification
        self.google_search_console_verification = config.get('google_search_console_key')
    
    def generate_global_meta_tags(self) -> str:
        """Generate global meta tags for all pages with proper AdSense verification"""
        meta_tags = ""
        
        # CRITICAL: AdSense verification meta tag - MUST be on ALL pages
        if self.google_adsense_id:
            # Ensure proper format
            adsense_id = self.google_adsense_id
            if not adsense_id.startswith('ca-pub-'):
                adsense_id = f"ca-pub-{adsense_id}"
            
            meta_tags += f'    <meta name="google-adsense-account" content="{adsense_id}">\n'
        
        # Google Search Console verification
        if self.google_search_console_verification:
            meta_tags += f'    <meta name="google-site-verification" content="{self.google_search_console_verification}">\n'
        
        # Google Analytics with proper error handling
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
        
        # AdSense script - CRITICAL: Must load without error suppression
        if self.google_adsense_id:
            adsense_id = self.google_adsense_id
            if not adsense_id.startswith('ca-pub-'):
                adsense_id = f"ca-pub-{adsense_id}"
            
            meta_tags += f'''    <!-- Google AdSense -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={adsense_id}" 
            crossorigin="anonymous"></script>
'''
        
        return meta_tags
    
    def generate_meta_tags(self, post) -> str:
        """Generate comprehensive meta tags for individual posts"""
        base_path = self.config.get("base_path", "")
        post_url = f"{self.config['base_url']}{base_path}/{post.slug}/"
        
        # Get featured image if available
        image_url = None
        if hasattr(post, 'featured_image') and post.featured_image:
            image_url = post.featured_image
        elif hasattr(post, 'image') and post.image:
            image_url = post.image
        
        meta_tags = f'''
    <!-- SEO Meta Tags -->
    <meta name="description" content="{post.meta_description}">
    <meta property="og:title" content="{post.title}">
    <meta property="og:description" content="{post.meta_description}">
    <meta property="og:url" content="{post_url}">
    <meta property="og:type" content="article">
    <meta property="og:site_name" content="{self.config['site_name']}">
    <meta property="article:published_time" content="{post.created_at}">
    <meta property="article:modified_time" content="{post.updated_at}">'''
        
        # Add image meta tags if available
        if image_url:
            meta_tags += f'''
    <meta property="og:image" content="{image_url}">
    <meta property="og:image:alt" content="{post.title}">
    <meta name="twitter:image" content="{image_url}">'''

        meta_tags += f'''

    <!-- Twitter Cards -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{post.title}">
    <meta name="twitter:description" content="{post.meta_description}">'''
        
        # Add Twitter handle if available
        if self.config.get('social_accounts', {}).get('twitter'):
            twitter_handle = self.config['social_accounts']['twitter'].replace('@', '')
            meta_tags += f'''
    <meta name="twitter:site" content="@{twitter_handle}">
    <meta name="twitter:creator" content="@{twitter_handle}">'''

        meta_tags += f'''

    <!-- Additional SEO -->
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1">
    <meta name="googlebot" content="index, follow">
    <link rel="canonical" href="{post_url}">'''
        
        # Add keywords if available
        if hasattr(post, 'seo_keywords') and post.seo_keywords:
            keywords_str = ", ".join(post.seo_keywords) if isinstance(post.seo_keywords, list) else post.seo_keywords
            meta_tags += f'\n    <meta name="keywords" content="{keywords_str}">'
        
        return meta_tags
    
    def generate_adsense_ad(self, slot_type: str = "display", slot_id: str = None) -> str:
        """Generate AdSense ad unit HTML with proper implementation"""
        if not self.google_adsense_id:
            return f'<div class="ad-placeholder ad-{slot_type}"><!-- AdSense Ad Slot ({slot_type}) --></div>'
        
        # Ensure proper AdSense ID format
        adsense_id = self.google_adsense_id
        if not adsense_id.startswith('ca-pub-'):
            adsense_id = f"ca-pub-{adsense_id}"
        
        # Use provided slot ID or auto ads
        data_ad_slot = f'data-ad-slot="{slot_id}"' if slot_id else ''
        
        # Different ad formats based on slot type
        if slot_type == "header":
            ad_format = 'data-ad-format="leaderboard"'
            style = "display:inline-block;width:728px;height:90px"
        elif slot_type == "middle":
            ad_format = 'data-ad-format="rectangle"'
            style = "display:inline-block;width:300px;height:250px"
        elif slot_type == "footer":
            ad_format = 'data-ad-format="banner"'
            style = "display:inline-block;width:468px;height:60px"
        else:
            ad_format = 'data-ad-format="auto"'
            style = "display:block"
        
        # For responsive design
        responsive = 'data-full-width-responsive="true"' if not slot_id else ''
        
        ad_html = f'''<div class="ad-{slot_type}" style="text-align: center; margin: 20px 0;">
    <ins class="adsbygoogle"
         style="{style}"
         data-ad-client="{adsense_id}"
         {data_ad_slot}
         {ad_format}
         {responsive}></ins>
    <script>
        (adsbygoogle = window.adsbygoogle || []).push({{}});
    </script>
</div>'''
        
        return ad_html

    def generate_structured_data(self, post) -> str:
        """Generate JSON-LD structured data for better SEO"""
        base_path = self.config.get("base_path", "")
        
        # Add image if available
        image_url = None
        if hasattr(post, 'featured_image') and post.featured_image:
            image_url = post.featured_image
        elif hasattr(post, 'image') and post.image:
            image_url = post.image
        
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
                "url": f"{self.config['base_url']}{base_path}",
                "logo": {
                    "@type": "ImageObject",
                    "url": f"{self.config['base_url']}{base_path}/static/logo.png"
                }
            },
            "datePublished": post.created_at,
            "dateModified": post.updated_at,
            "url": f"{self.config['base_url']}{base_path}/{post.slug}/",
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": f"{self.config['base_url']}{base_path}/{post.slug}/"
            }
        }
        
        # Add image if available
        if image_url:
            structured_data["image"] = {
                "@type": "ImageObject",
                "url": image_url
            }
        
        # Add keywords if available
        if hasattr(post, 'seo_keywords') and post.seo_keywords:
            structured_data["keywords"] = post.seo_keywords
        
        return f'<script type="application/ld+json">\n{json.dumps(structured_data, indent=2)}\n</script>'

    def generate_sitemap(self, posts) -> str:
        """Generate sitemap with base_path support and better formatting"""
        base_path = self.config.get("base_path", "")
        
        urls = []
        
        # Add homepage
        urls.append(f'''    <url>
        <loc>{self.config['base_url']}{base_path}/</loc>
        <lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
    </url>''')
        
        # Add about page
        urls.append(f'''    <url>
        <loc>{self.config['base_url']}{base_path}/about/</loc>
        <lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.8</priority>
    </url>''')
        
        # Add posts
        for post in posts:
            last_modified = post.updated_at.split('T')[0] if 'T' in post.updated_at else post.updated_at
            urls.append(f'''    <url>
        <loc>{self.config['base_url']}{base_path}/{post.slug}/</loc>
        <lastmod>{last_modified}</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.9</priority>
    </url>''')
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{''.join(urls)}
</urlset>"""

    def generate_robots_txt(self) -> str:
        """Generate robots.txt with base_path support"""
        base_path = self.config.get("base_path", "")
        
        # Fix: Don't add base_path if base_url already includes it
        if base_path and self.config['base_url'].endswith(base_path):
            sitemap_url = f"{self.config['base_url']}/sitemap.xml"
            rss_url = f"{self.config['base_url']}/rss.xml"
        else:
            sitemap_url = f"{self.config['base_url']}{base_path}/sitemap.xml"
            rss_url = f"{self.config['base_url']}{base_path}/rss.xml"
        
        robots_content = f"""User-agent: *
Allow: /
Disallow: /static/admin/
Disallow: /static/temp/
Disallow: /*.json$
Disallow: /*?*

# Crawl-delay for polite crawling
Crawl-delay: 1

# Sitemap location
Sitemap: {sitemap_url}

# RSS Feed location
Sitemap: {rss_url}"""
        
        return robots_content

    def generate_homepage_meta_tags(self) -> str:
        """Generate meta tags specifically for the homepage"""
        base_path = self.config.get("base_path", "")
        homepage_url = f"{self.config['base_url']}{base_path}/"
        
        meta_tags = f'''
    <!-- Homepage SEO Meta Tags -->
    <meta name="description" content="{self.config['site_description']}">
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
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1">
    <meta name="googlebot" content="index, follow">
    <link rel="canonical" href="{homepage_url}">'''
        
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
            "logo": {
                "@type": "ImageObject",
                "url": f"{self.config['base_url']}{base_path}/static/logo.png"
            }
        }
        
        if social_urls:
            organization_data["sameAs"] = social_urls
        
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
            "publisher": {
                "@type": "Organization",
                "name": self.config["site_name"]
            },
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
            
            # Handle datetime parsing more safely
            try:
                if 'T' in post.created_at:
                    pub_date = datetime.fromisoformat(post.created_at.replace('Z', '+00:00')).strftime('%a, %d %b %Y %H:%M:%S %z')
                else:
                    pub_date = datetime.strptime(post.created_at, '%Y-%m-%d').strftime('%a, %d %b %Y %H:%M:%S %z')
            except:
                pub_date = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
            
            rss_items.append(f"""
    <item>
        <title><![CDATA[{post.title}]]></title>
        <description><![CDATA[{post.meta_description}]]></description>
        <link>{post_url}</link>
        <guid isPermaLink="true">{post_url}</guid>
        <pubDate>{pub_date}</pubDate>
    </item>""")
        
        current_date = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
    <channel>
        <title>{self.config['site_name']}</title>
        <description>{self.config['site_description']}</description>
        <link>{self.config['base_url']}{base_path}/</link>
        <atom:link href="{self.config['base_url']}{base_path}/rss.xml" rel="self" type="application/rss+xml" />
        <language>en-us</language>
        <lastBuildDate>{current_date}</lastBuildDate>
        <generator>AI Blog System</generator>
        {''.join(rss_items)}
    </channel>
</rss>"""

    def validate_adsense_setup(self) -> dict:
        """Validate AdSense configuration for approval"""
        issues = []
        warnings = []
        
        # Check AdSense ID
        if not self.google_adsense_id:
            issues.append("No Google AdSense ID configured")
        elif not self.google_adsense_id.startswith('ca-pub-'):
            if not self.google_adsense_id.replace('ca-pub-', '').isdigit():
                issues.append("AdSense ID format incorrect - should be numbers only after 'ca-pub-'")
        
        # Check required pages
        required_pages = ['Privacy Policy', 'Terms of Service', 'About']
        for page in required_pages:
            warnings.append(f"Ensure {page} page exists and is accessible")
        
        # Check content requirements
        warnings.extend([
            "Ensure site has 20+ high-quality, original posts",
            "Verify site has decent organic traffic (1000+ monthly visitors recommended)",
            "Make sure content is in a supported language",
            "Ensure site navigation is clear and functional",
            "Verify all links work properly",
            "Check that site is mobile-friendly",
            "Ensure HTTPS is enabled (GitHub Pages provides this)"
        ])
        
        return {
            'issues': issues,
            'warnings': warnings,
            'adsense_configured': bool(self.google_adsense_id),
            'ready_for_approval': len(issues) == 0
        }