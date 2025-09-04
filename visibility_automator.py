import requests
import json
import base64
import hashlib
import hmac
import time
from datetime import datetime
from urllib.parse import quote, urlencode
from typing import List, Dict, Optional
import logging

class TwitterAPI:
    """Twitter API v2 integration for posting tweets"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str, bearer_token: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2/"
        
    def _generate_oauth_signature(self, method: str, url: str, params: dict) -> str:
        """Generate OAuth 1.0a signature"""
        # Create parameter string
        sorted_params = sorted(params.items())
        param_string = '&'.join([f"{quote(str(k))}={quote(str(v))}" for k, v in sorted_params])
        
        # Create signature base string
        base_string = f"{method.upper()}&{quote(url)}&{quote(param_string)}"
        
        # Create signing key
        signing_key = f"{quote(self.api_secret)}&{quote(self.access_token_secret)}"
        
        # Generate signature
        signature = base64.b64encode(
            hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
        ).decode()
        
        return signature
    
    def _create_oauth_header(self, method: str, url: str, params: dict) -> str:
        """Create OAuth 1.0a authorization header"""
        oauth_params = {
            'oauth_consumer_key': self.api_key,
            'oauth_nonce': str(int(time.time() * 1000)),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_token': self.access_token,
            'oauth_version': '1.0'
        }
        
        # Add request parameters for signature calculation
        all_params = {**oauth_params, **params}
        oauth_params['oauth_signature'] = self._generate_oauth_signature(method, url, all_params)
        
        # Create authorization header
        oauth_header = 'OAuth ' + ', '.join([f'{k}="{quote(str(v))}"' for k, v in sorted(oauth_params.items())])
        return oauth_header
    
    def post_tweet(self, text: str, media_ids: List[str] = None) -> dict:
        """Post a tweet using Twitter API v2"""
        url = "https://api.twitter.com/2/tweets"
        
        tweet_data = {
            "text": text
        }
        
        if media_ids:
            tweet_data["media"] = {"media_ids": media_ids}
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=tweet_data, headers=headers, timeout=30)
            
            if response.status_code == 201:
                return {
                    "success": True,
                    "data": response.json(),
                    "tweet_id": response.json().get('data', {}).get('id')
                }
            else:
                return {
                    "success": False,
                    "error": response.json(),
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_media(self, file_path: str, media_category: str = "tweet_image") -> Optional[str]:
        """Upload media to Twitter and return media_id"""
        # This is a simplified version - you'd need to implement chunked upload for larger files
        url = "https://upload.twitter.com/1.1/media/upload.json"
        
        try:
            with open(file_path, 'rb') as file:
                files = {'media': file}
                data = {'media_category': media_category}
                
                # Create OAuth header for media upload
                oauth_header = self._create_oauth_header('POST', url, {})
                headers = {'Authorization': oauth_header}
                
                response = requests.post(url, files=files, data=data, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    return response.json().get('media_id_string')
                else:
                    logging.error(f"Media upload failed: {response.json()}")
                    return None
                    
        except Exception as e:
            logging.error(f"Media upload error: {str(e)}")
            return None

class VisibilityAutomator:
    """Enhanced content distribution and visibility automation with Twitter integration"""
    
    def __init__(self, config):
        self.config = config
        self.social_accounts = config.get('social_accounts', {})
        self.twitter_config = config.get('twitter_api', {})
        self.twitter_api = None
        
        # Initialize Twitter API if credentials are provided
        if all(key in self.twitter_config for key in ['api_key', 'api_secret', 'access_token', 'access_token_secret', 'bearer_token']):
            self.twitter_api = TwitterAPI(
                self.twitter_config['api_key'],
                self.twitter_config['api_secret'],
                self.twitter_config['access_token'],
                self.twitter_config['access_token_secret'],
                self.twitter_config['bearer_token']
            )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def submit_to_search_engines(self, sitemap_url: str) -> List[dict]:
        """Submit sitemap to search engines"""
        search_engines = [
            f"https://www.google.com/ping?sitemap={quote(sitemap_url)}",
            f"https://www.bing.com/ping?sitemap={quote(sitemap_url)}"
        ]
        
        results = []
        for engine in search_engines:
            try:
                response = requests.get(engine, timeout=10)
                engine_name = engine.split('//')[1].split('.')[1]
                results.append({
                    'engine': engine_name,
                    'status': response.status_code,
                    'success': response.status_code == 200,
                    'url': engine
                })
                self.logger.info(f"Submitted to {engine_name}: {response.status_code}")
            except Exception as e:
                engine_name = engine.split('//')[1].split('.')[1]
                results.append({
                    'engine': engine_name,
                    'error': str(e),
                    'success': False
                })
                self.logger.error(f"Failed to submit to {engine_name}: {str(e)}")
        
        return results
    
    def generate_social_posts(self, post) -> dict:
        """Generate optimized social media posts for different platforms"""
        base_path = self.config.get("base_path", "")
        post_url = f"{self.config['base_url']}{base_path}/{post.slug}/"
        
        # Generate hashtags from tags or keywords
        hashtags = []
        if hasattr(post, 'tags') and post.tags:
            hashtags = ['#' + tag.replace(' ', '').replace('-', '').title() 
                       for tag in post.tags[:3]]
        elif hasattr(post, 'seo_keywords') and post.seo_keywords:
            keywords = post.seo_keywords.split(',')[:3]
            hashtags = ['#' + kw.strip().replace(' ', '').title() for kw in keywords]
        
        # Add site-specific hashtag if configured
        if self.config.get('default_hashtag'):
            hashtags.append(f"#{self.config['default_hashtag']}")
        
        social_posts = {
            'twitter': self._generate_twitter_post(post, post_url, hashtags),
            'linkedin': self._generate_linkedin_post(post, post_url, hashtags),
            'facebook': self._generate_facebook_post(post, post_url),
            'reddit': {
                'title': post.title,
                'content': f"{post.meta_description}\n\nFull article: {post_url}"
            }
        }
        
        return social_posts
    
    def _generate_twitter_post(self, post, post_url: str, hashtags: List[str]) -> str:
        """Generate Twitter-optimized post with character limit consideration"""
        max_length = 280
        url_length = 23  # Twitter's t.co URL length
        hashtag_text = ' '.join(hashtags[:2])  # Limit hashtags for Twitter
        
        # Calculate available space
        available_space = max_length - url_length - len(hashtag_text) - 3  # 3 for spaces
        
        # Create intro text
        if len(post.title) + len(post.meta_description) + 10 < available_space:
            intro = f"📝 {post.title}\n\n{post.meta_description[:100]}{'...' if len(post.meta_description) > 100 else ''}"
        else:
            intro = f"📝 {post.title}"
            remaining = available_space - len(intro) - 2
            if remaining > 20:
                intro += f"\n\n{post.meta_description[:remaining]}..."
        
        return f"{intro}\n\n{post_url} {hashtag_text}".strip()
    
    def _generate_linkedin_post(self, post, post_url: str, hashtags: List[str]) -> str:
        """Generate LinkedIn-optimized post"""
        intro = f"🚀 New blog post: {post.title}\n\n{post.meta_description}"
        
        # Add a call to action
        cta = "\n\n💭 What are your thoughts on this topic?"
        hashtag_text = '\n\n' + ' '.join(hashtags)
        
        return f"{intro}\n\n📖 Read the full article: {post_url}{cta}{hashtag_text}"
    
    def _generate_facebook_post(self, post, post_url: str) -> str:
        """Generate Facebook-optimized post"""
        return f"📝 {post.title}\n\n{post.meta_description}\n\n🔗 Read more: {post_url}"
    
    def post_to_twitter(self, post, custom_text: str = None, image_path: str = None) -> dict:
        """Post blog intro to Twitter"""
        if not self.twitter_api:
            return {
                "success": False,
                "error": "Twitter API not configured. Please provide API credentials in config."
            }
        
        try:
            # Use custom text or generate social post
            if custom_text:
                tweet_text = custom_text
            else:
                social_posts = self.generate_social_posts(post)
                tweet_text = social_posts['twitter']
            
            # Upload image if provided
            media_ids = None
            if image_path:
                media_id = self.twitter_api.upload_media(image_path)
                if media_id:
                    media_ids = [media_id]
                    self.logger.info(f"Media uploaded successfully: {media_id}")
                else:
                    self.logger.warning("Failed to upload media, posting without image")
            
            # Post the tweet
            result = self.twitter_api.post_tweet(tweet_text, media_ids)
            
            if result['success']:
                self.logger.info(f"Tweet posted successfully: {result.get('tweet_id')}")
                return {
                    "success": True,
                    "tweet_id": result.get('tweet_id'),
                    "text": tweet_text,
                    "url": f"https://twitter.com/user/status/{result.get('tweet_id')}" if result.get('tweet_id') else None
                }
            else:
                self.logger.error(f"Failed to post tweet: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get('error'),
                    "text": tweet_text
                }
        
        except Exception as e:
            self.logger.error(f"Twitter posting error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def auto_post_new_blog(self, post, platforms: List[str] = None, delay_minutes: int = 0) -> dict:
        """Automatically post to specified social platforms when new blog is published"""
        if platforms is None:
            platforms = ['twitter']  # Default to Twitter only
        
        results = {}
        
        # Add delay if specified (useful for scheduling)
        if delay_minutes > 0:
            import time
            self.logger.info(f"Waiting {delay_minutes} minutes before posting...")
            time.sleep(delay_minutes * 60)
        
        # Post to Twitter
        if 'twitter' in platforms:
            twitter_result = self.post_to_twitter(post)
            results['twitter'] = twitter_result
        
        # Add other platforms here as needed
        # if 'linkedin' in platforms:
        #     results['linkedin'] = self.post_to_linkedin(post)
        
        return results
    
    def schedule_social_posts(self, posts: List, interval_hours: int = 2) -> List[dict]:
        """Schedule social media posts for multiple blog posts"""
        import threading
        import time
        
        scheduled_posts = []
        
        for i, post in enumerate(posts):
            delay_seconds = i * interval_hours * 3600
            
            def post_delayed(p=post, delay=delay_seconds):
                time.sleep(delay)
                return self.post_to_twitter(p)
            
            # Start thread for delayed posting
            thread = threading.Thread(target=post_delayed)
            thread.start()
            
            scheduled_posts.append({
                'post': post.title,
                'scheduled_time': datetime.now().timestamp() + delay_seconds,
                'thread': thread
            })
        
        return scheduled_posts
    
    def get_posting_analytics(self, post_results: List[dict]) -> dict:
        """Analyze posting results and provide insights"""
        total_posts = len(post_results)
        successful_posts = sum(1 for result in post_results if result.get('success'))
        failed_posts = total_posts - successful_posts
        
        analytics = {
            'total_attempted': total_posts,
            'successful': successful_posts,
            'failed': failed_posts,
            'success_rate': successful_posts / total_posts if total_posts > 0 else 0,
            'platforms_used': [],
            'errors': []
        }
        
        # Collect platform info and errors
        for result in post_results:
            if 'platform' in result:
                analytics['platforms_used'].append(result['platform'])
            if not result.get('success') and 'error' in result:
                analytics['errors'].append(result['error'])
        
        return analytics
    
    def validate_twitter_config(self) -> dict:
        """Validate Twitter API configuration"""
        required_keys = ['api_key', 'api_secret', 'access_token', 'access_token_secret', 'bearer_token']
        missing_keys = [key for key in required_keys if not self.twitter_config.get(key)]
        
        if missing_keys:
            return {
                'valid': False,
                'missing_keys': missing_keys,
                'message': f"Missing Twitter API keys: {', '.join(missing_keys)}"
            }
        
        # Test API connection if all keys are present
        if self.twitter_api:
            try:
                # Test with a simple API call (you might want to implement a test endpoint)
                return {
                    'valid': True,
                    'message': "Twitter API configuration is valid"
                }
            except Exception as e:
                return {
                    'valid': False,
                    'error': str(e),
                    'message': "Twitter API keys provided but connection failed"
                }
        
        return {
            'valid': False,
            'message': "Twitter API not initialized"
        }
    
    def create_content_calendar(self, posts: List, start_date: datetime = None, posts_per_week: int = 3) -> List[dict]:
        """Create a content calendar for social media posts"""
        if not start_date:
            start_date = datetime.now()
        
        calendar = []
        days_between_posts = 7 / posts_per_week  # Days between each post
        
        for i, post in enumerate(posts):
            post_date = start_date + datetime.timedelta(days=i * days_between_posts)
            
            calendar_entry = {
                'post_title': post.title,
                'post_slug': post.slug,
                'scheduled_date': post_date.isoformat(),
                'day_of_week': post_date.strftime('%A'),
                'platforms': ['twitter'],  # Default platforms
                'status': 'scheduled'
            }
            
            calendar.append(calendar_entry)
        
        return calendar
    
    def _format_rss_date(self, iso_date: str) -> str:
        """Convert ISO date to RSS format"""
        try:
            if 'T' in iso_date:
                dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(iso_date, '%Y-%m-%d')
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')
        except:
            dt = datetime.now()
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')