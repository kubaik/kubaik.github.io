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
        # Fixed: Google Search Console uses verification key, not API key
        self.google_search_console_verification = config.get('google_search_console_verification')
    
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
                # Add logo for better structured data
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
    
    def generate_global_meta_tags(self) -> str:
        """Generate global meta tags for all pages"""
        meta_tags = ""
        
        # Add AdSense verification if available
        if self.google_adsense_verification:
            meta_tags += f'    <meta name="google-adsense-account" content="{self.google_adsense_verification}">\n'
        
        # Fixed: Use correct Google Search Console verification
        if self.google_search_console_verification:
            meta_tags += f'    <meta name="google-site-verification" content="{self.google_search_console_verification}">\n'
        
        # Add other global verification codes
        if self.config.get('verification_code'):
            meta_tags += f'    <meta name="verification" content="{self.config["verification_code"]}">\n'
        
        # Fixed: Improved Google Analytics implementation
        if self.google_analytics_id:
            meta_tags += f'''    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={self.google_analytics_id}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{self.google_analytics_id}', {{
            page_title: document.title,
            page_location: window.location.href
        }});
    </script>
'''
        
        # Fixed: Improved AdSense implementation with error handling
        if self.google_adsense_id:
            meta_tags += f'''    <!-- Google AdSense -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={self.google_adsense_id}" 
            crossorigin="anonymous" 
            onerror="console.warn('AdSense failed to load')"></script>
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
            meta_tags += f'\n    <meta name="keywords" content="{post.seo_keywords}">'
        
        return meta_tags
    
    def generate_adsense_ad(self, slot_type: str = "display", slot_id: str = None) -> str:
        """Generate AdSense ad unit HTML with better error handling"""
        if not self.google_adsense_id:
            return '<div class="ad-placeholder" style="min-height: 250px; background: #f5f5f5; display: flex; align-items: center; justify-content: center; color: #999;"><!-- AdSense Ad Slot --></div>'
        
        # Use provided slot ID or generate auto slot
        data_ad_slot = f'data-ad-slot="{slot_id}"' if slot_id else 'data-ad-slot="AUTO"'
        
        ad_html = f'''
<div class="ad-container ad-{slot_type}" style="text-align: center; margin: 20px 0;">
    <ins class="adsbygoogle"
         style="display:block"
         data-ad-client="{self.google_adsense_id}"
         {data_ad_slot}
         data-ad-format="auto"
         data-full-width-responsive="true"></ins>
    <script>
        try {{
            (adsbygoogle = window.adsbygoogle || []).push({{}});
        }} catch (e) {{
            console.warn('AdSense initialization failed:', e);
        }}
    </script>
</div>'''
        
        return ad_html

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
        
        robots_content = f"""User-agent: *
Allow: /
Disallow: /static/admin/
Disallow: /static/temp/
Disallow: /*.json$
Disallow: /*?*

# Crawl-delay for polite crawling
Crawl-delay: 1

# Sitemap location
Sitemap: {self.config['base_url']}{base_path}/sitemap.xml

# RSS Feed location
Sitemap: {self.config['base_url']}{base_path}/rss.xml"""
        
        return robots_content

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
                    # Assume it's already in a readable format
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
            
            # Add width and height if not present (helps with CLS)
            if 'width=' not in attrs and 'height=' not in attrs:
                attrs += ' style="max-width: 100%; height: auto;"'
            
            return f'<img{attrs}>'
        
        return re.sub(img_pattern, add_attributes, content)

    def generate_seo_report(self, posts) -> dict:
        """Generate SEO analysis report"""
        if not posts:
            return {
                'total_posts': 0,
                'recommendations': ['No posts found to analyze']
            }
            
        report = {
            'total_posts': len(posts),
            'posts_with_meta_description': 0,
            'posts_with_keywords': 0,
            'posts_with_images': 0,
            'average_title_length': 0,
            'average_meta_description_length': 0,
            'title_issues': [],
            'meta_description_issues': [],
            'recommendations': []
        }
        
        total_title_length = 0
        total_meta_length = 0
        
        for i, post in enumerate(posts):
            # Check meta descriptions
            if hasattr(post, 'meta_description') and post.meta_description:
                report['posts_with_meta_description'] += 1
                meta_len = len(post.meta_description)
                total_meta_length += meta_len
                
                # Check meta description length
                if meta_len < 120:
                    report['meta_description_issues'].append(f"Post '{post.title}': Meta description too short ({meta_len} chars)")
                elif meta_len > 160:
                    report['meta_description_issues'].append(f"Post '{post.title}': Meta description too long ({meta_len} chars)")
            
            # Check keywords
            if hasattr(post, 'seo_keywords') and post.seo_keywords:
                report['posts_with_keywords'] += 1
            
            # Check images
            if hasattr(post, 'featured_image') and post.featured_image:
                report['posts_with_images'] += 1
            elif hasattr(post, 'image') and post.image:
                report['posts_with_images'] += 1
            
            # Check title length
            if hasattr(post, 'title'):
                title_len = len(post.title)
                total_title_length += title_len
                
                if title_len > 60:
                    report['title_issues'].append(f"Post '{post.title}': Title too long ({title_len} chars)")
                elif title_len < 30:
                    report['title_issues'].append(f"Post '{post.title}': Title too short ({title_len} chars)")
        
        # Calculate averages
        report['average_title_length'] = total_title_length / len(posts)
        if report['posts_with_meta_description'] > 0:
            report['average_meta_description_length'] = total_meta_length / report['posts_with_meta_description']
        
        # Generate recommendations
        meta_desc_percentage = report['posts_with_meta_description'] / len(posts)
        if meta_desc_percentage < 0.8:
            report['recommendations'].append(f"Add meta descriptions to more posts ({meta_desc_percentage:.1%} have them)")
        
        keywords_percentage = report['posts_with_keywords'] / len(posts)
        if keywords_percentage < 0.5:
            report['recommendations'].append(f"Add SEO keywords to more posts ({keywords_percentage:.1%} have them)")
        
        images_percentage = report['posts_with_images'] / len(posts)
        if images_percentage < 0.7:
            report['recommendations'].append(f"Add featured images to more posts ({images_percentage:.1%} have them)")
        
        if report['average_title_length'] > 60:
            report['recommendations'].append("Consider shorter post titles for better SEO (average: {:.0f} chars)".format(report['average_title_length']))
        
        if not report['recommendations']:
            report['recommendations'].append("SEO looks good! Keep up the great work.")
        
        return report

    def generate_performance_hints(self) -> str:
        """Generate performance optimization hints for HTML"""
        return '''
    <!-- Performance Optimization -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="preconnect" href="https://www.google-analytics.com">
    <link rel="preconnect" href="https://pagead2.googlesyndication.com">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#ffffff">'''

    def validate_config(self) -> list:
        """Validate SEO configuration and return warnings/errors"""
        warnings = []
        
        # Check required fields
        required_fields = ['site_name', 'site_description', 'base_url']
        for field in required_fields:
            if not self.config.get(field):
                warnings.append(f"Missing required field: {field}")
        
        # Check Google Analytics ID format
        if self.google_analytics_id and not self.google_analytics_id.startswith('G-'):
            warnings.append("Google Analytics ID should start with 'G-'")
        
        # Check AdSense ID format
        if self.google_adsense_id and not self.google_adsense_id.startswith('ca-pub-'):
            warnings.append("Google AdSense ID should start with 'ca-pub-'")
        
        # Check social accounts
        social_accounts = self.config.get('social_accounts', {})
        if social_accounts.get('twitter') and not social_accounts['twitter'].startswith('@'):
            warnings.append("Twitter handle should start with '@'")
        
        return warnings