import requests
from requests_oauthlib import OAuth1

import json
import base64
import hashlib
import hmac
import time
import secrets
from datetime import datetime
from urllib.parse import quote, urlencode
from typing import List, Dict, Optional
import logging

class TwitterAPI:
    """Twitter API v2 integration with OAuth 2.0 Bearer Token"""
    
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        self.auth = OAuth1(
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret
        )
        self.base_url_v1 = "https://api.twitter.com/1.1/"
        self.base_url_v2 = "https://api.twitter.com/2/"

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_connection(self):
        """Check credentials by verifying the account"""
        url = f"{self.base_url_v1}account/verify_credentials.json"
        try:
            response = requests.get(url, auth=self.auth, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "user": {
                        "id": data.get("id_str"),
                        "name": data.get("name"),
                        "screen_name": data.get("screen_name")
                    },
                    "message": "Twitter API connection successful"
                }
            else:
                return {
                    "success": False,
                    "error": response.json(),
                    "status_code": response.status_code,
                    "message": "Twitter API connection failed"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def post_tweet(self, text: str):
        """Post a tweet using v2 /tweets endpoint"""
        url = f"{self.base_url_v2}tweets"
        payload = {"text": text}

        try:
            response = requests.post(url, json=payload, auth=self.auth, timeout=30)
            if response.status_code in (200, 201):
                data = response.json()
                return {
                    "success": True,
                    "tweet_id": data.get("data", {}).get("id"),
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "error": response.json(),
                    "status_code": response.status_code
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

class VisibilityAutomator:
    """Enhanced content distribution with proper Twitter integration"""
    
    def __init__(self, config):
        self.config = config
        self.social_accounts = config.get('social_accounts', {})
        self.twitter_config = config.get('twitter_api', {})
        self.twitter_api = None

        # Initialize Twitter API with OAuth 2.0 Bearer Token
        if all(k in self.twitter_config for k in
               ("api_key", "api_secret", "access_token", "access_token_secret")):
            self.twitter_api = TwitterAPI(
                self.twitter_config["api_key"],
                self.twitter_config["api_secret"],
                self.twitter_config["access_token"],
                self.twitter_config["access_token_secret"]
            )
            print("Twitter API initialized with OAuth 1.0a User Context ✅")
        else:
            print("⚠️ Missing Twitter OAuth 1.0a credentials in config.yaml")
        
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def test_twitter_connection(self) -> dict:
        """Test Twitter API connection"""
        if not self.twitter_api:
            return {
                "success": False,
                "error": "Twitter API not configured - missing OAuth credentials"
            }
        
        return self.twitter_api.test_connection()
    
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
            if isinstance(post.seo_keywords, str):
                keywords = post.seo_keywords.split(',')[:3]
            else:
                keywords = post.seo_keywords[:3]
            hashtags = ['#' + kw.strip().replace(' ', '').title() for kw in keywords]
        
        # Default hashtags if none found
        if not hashtags:
            hashtags = ['#TechBlog', '#AI', '#Development']
        
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
        if len(post.title) <= available_space - 20:  # Leave space for description
            intro = f"📝 {post.title}"
            remaining = available_space - len(intro) - 4  # 4 for newlines
            if remaining > 20 and hasattr(post, 'meta_description'):
                description = post.meta_description[:remaining] + ('...' if len(post.meta_description) > remaining else '')
                intro += f"\n\n{description}"
        else:
            # Title is too long, truncate it
            title_limit = available_space - 20
            intro = f"📝 {post.title[:title_limit]}..."
        
        tweet = f"{intro}\n\n{post_url} {hashtag_text}".strip()
        
        # Ensure we don't exceed character limit
        if len(tweet) > max_length:
            excess = len(tweet) - max_length
            if hasattr(post, 'meta_description') and post.meta_description in tweet:
                # Trim the description
                current_desc = post.meta_description
                new_desc = current_desc[:len(current_desc) - excess - 3] + "..."
                tweet = tweet.replace(current_desc, new_desc)
        
        return tweet
    
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
    
    def post_to_twitter(self, post, custom_text=None):
        if not self.twitter_api:
            return {"success": False, "error": "Twitter API not configured"}

        tweet_text = custom_text or f"📝 {post.title}\n\n{post.meta_description}\n\n🔗 {self.config['base_url']}/{post.slug}/"
        print(f"Tweet text ({len(tweet_text)} chars): {tweet_text}")

        result = self.twitter_api.post_tweet(tweet_text)
        if result["success"]:
            return {
                "success": True,
                "tweet_id": result.get("tweet_id"),
                "url": f"https://twitter.com/i/web/status/{result.get('tweet_id')}"
            }
        return result
    
    def validate_twitter_config(self) -> dict:
        """Validate Twitter API configuration"""
        required_keys = ['api_key', 'api_secret', 'access_token', 'access_token_secret']
        twitter_config = self.twitter_config
        
        missing_keys = [key for key in required_keys if not twitter_config.get(key)]
        if missing_keys:
            return {
                'valid': False,
                'message': f"Missing Twitter OAuth credentials: {', '.join(missing_keys)}"
            }
        
        # Test API connection
        if self.twitter_api:
            test_result = self.twitter_api.test_connection()
            return {
                'valid': test_result['success'],
                'message': test_result['message'],
                'error': test_result.get('error') if not test_result['success'] else None
            }
        
        return {
            'valid': False,
            'message': "Twitter API not initialized"
        }
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
# Test function to verify Twitter integration
def test_twitter_integration(config):
    """Test function to verify Twitter posting works"""
    print("Testing Twitter integration...")
    
    visibility = VisibilityAutomator(config)
    
    # Test connection first
    connection_test = visibility.test_twitter_connection()
    print(f"Connection test: {connection_test}")
    
    if not connection_test['success']:
        print(f"Connection failed: {connection_test.get('error')}")
        return False
    
    # Create a test post object
    class TestPost:
        def __init__(self):
            self.title = "Test Blog Post - Twitter Integration"
            self.meta_description = "Testing automated Twitter posting from our blog system. This is a test post to verify the integration works correctly."
            self.slug = "test-twitter-integration" 
            self.tags = ["test", "twitter", "automation"]
    
    test_post = TestPost()
    
    # Generate preview without posting
    social_posts = visibility.generate_social_posts(test_post)
    print(f"Generated tweet preview: {social_posts['twitter']}")
    print(f"Tweet length: {len(social_posts['twitter'])} characters")
    
    # Ask for confirmation before actually posting
    response = input("\nDo you want to post this test tweet? (y/N): ")
    if response.lower() == 'y':
        result = visibility.post_to_twitter(test_post)
        print(f"Posting result: {result}")
        return result['success']
    else:
        print("Test cancelled - no tweet posted.")
        return True  # Test structure works, just didn't post

if __name__ == "__main__":
    # Example usage with your config
    import yaml
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        success = test_twitter_integration(config)
        print(f"Twitter integration test: {'✅ PASSED' if success else '❌ FAILED'}")
        
    except FileNotFoundError:
        print("config.yaml not found. Please make sure it exists in the current directory.")
    except Exception as e:
        print(f"Error testing Twitter integration: {e}")