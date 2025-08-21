import requests
from datetime import datetime
from urllib.parse import quote
from typing import List, Dict

class VisibilityAutomator:
    """Automates content distribution and visibility"""
    
    def __init__(self, config):
        self.config = config
        self.social_accounts = config.get('social_accounts', {})
    
    def submit_to_search_engines(self, sitemap_url: str):
        """Submit sitemap to search engines"""
        search_engines = [
            f"https://www.google.com/ping?sitemap={quote(sitemap_url)}",
            f"https://www.bing.com/ping?sitemap={quote(sitemap_url)}"
        ]
        
        results = []
        for engine in search_engines:
            try:
                response = requests.get(engine, timeout=10)
                results.append({
                    'engine': engine.split('//')[1].split('.')[1],
                    'status': response.status_code,
                    'success': response.status_code == 200
                })
            except Exception as e:
                results.append({
                    'engine': engine.split('//')[1].split('.')[1],
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def generate_social_posts(self, post) -> dict:
        """Generate social media posts for different platforms"""
        base_text = f"New blog post: {post.title}"
        post_url = f"{self.config['base_url']}/{post.slug}/"
        
        hashtags = ['#' + tag.replace(' ', '').replace('-', '') for tag in post.tags[:3]]
        
        social_posts = {
            'twitter': f"{base_text}\n\n{post.meta_description[:100]}...\n\n{post_url} {' '.join(hashtags[:2])}",
            'linkedin': f"{base_text}\n\n{post.meta_description}\n\nRead more: {post_url}\n\n{' '.join(hashtags)}",
            'facebook': f"{base_text}\n\n{post.meta_description}\n\n{post_url}",
            'reddit_title': post.title,
            'reddit_content': f"{post.meta_description}\n\nFull article: {post_url}"
        }
        
        return social_posts
    
    def create_rss_feed(self, posts: list) -> str:
        """Generate RSS feed for the blog"""
        rss_items = []
        for post in posts[:10]:  # Latest 10 posts
            rss_items.append(f'''
        <item>
            <title><![CDATA[{post.title}]]></title>
            <description><![CDATA[{post.meta_description}]]></description>
            <link>{self.config['base_url']}/{post.slug}/</link>
            <guid>{self.config['base_url']}/{post.slug}/</guid>
            <pubDate>{self._format_rss_date(post.created_at)}</pubDate>
        </item>''')
        
        rss_feed = f'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>{self.config['site_name']}</title>
        <description>{self.config['site_description']}</description>
        <link>{self.config['base_url']}</link>
        <lastBuildDate>{self._format_rss_date(datetime.now().isoformat())}</lastBuildDate>
        <language>en-US</language>
        {''.join(rss_items)}
    </channel>
</rss>'''
        
        return rss_feed
    
    def _format_rss_date(self, iso_date: str) -> str:
        """Convert ISO date to RSS format"""
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')
        except:
            dt = datetime.now()
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')