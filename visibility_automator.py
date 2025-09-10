import requests
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
    """Twitter API v2 integration with proper OAuth 1.0a authentication"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.base_url = "https://api.twitter.com/2/"
        
    def _generate_oauth_signature(self, method: str, url: str, params: dict) -> str:
        """Generate OAuth 1.0a signature"""
        # Create parameter string
        encoded_params = []
        for key in sorted(params.keys()):
            encoded_params.append(f"{quote(str(key))}={quote(str(params[key]))}")
        param_string = "&".join(encoded_params)
        
        # Create signature base string
        base_string = f"{method}&{quote(url)}&{quote(param_string)}"
        
        # Create signing key
        signing_key = f"{quote(self.api_secret)}&{quote(self.access_token_secret)}"
        
        # Generate signature
        signature = base64.b64encode(
            hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
        ).decode()
        
        return signature
    
    def _generate_oauth_header(self, method: str, url: str, additional_params: dict = None) -> str:
        """Generate OAuth 1.0a authorization header"""
        oauth_params = {
            'oauth_consumer_key': self.api_key,
            'oauth_token': self.access_token,
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_nonce': secrets.token_hex(16),
            'oauth_version': '1.0'
        }
        
        # Combine OAuth params with any additional params for signature
        all_params = oauth_params.copy()
        if additional_params:
            all_params.update(additional_params)
        
        # Generate signature
        oauth_params['oauth_signature'] = self._generate_oauth_signature(method, url, all_params)
        
        # Build authorization header
        auth_header_params = []
        for key in sorted(oauth_params.keys()):
            auth_header_params.append(f'{key}="{quote(str(oauth_params[key]))}"')
        
        return f"OAuth {', '.join(auth_header_params)}"
    
    def post_tweet(self, text: str) -> dict:
        """Post a tweet using Twitter API v2 with OAuth 1.0a"""
        url = "https://api.twitter.com/2/tweets"
        
        tweet_data = {
            "text": text
        }
        
        headers = {
            "Authorization": self._generate_oauth_header("POST", url),
            "Content-Type": "application/json"
        }
        
        try:
            print(f"Attempting to post tweet: {text[:50]}...")
            response = requests.post(url, json=tweet_data, headers=headers, timeout=30)
            
            print(f"Twitter API Response Status: {response.status_code}")
            print(f"Twitter API Response: {response.text}")
            
            if response.status_code == 201:
                response_data = response.json()
                return {
                    "success": True,
                    "data": response_data,
                    "tweet_id": response_data.get('data', {}).get('id')
                }
            else:
                return {
                    "success": False,
                    "error": response.json() if response.text else f"HTTP {response.status_code}",
                    "status_code": response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error: {str(e)}"
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def test_connection(self) -> dict:
        """Test the Twitter API connection using OAuth 1.0a"""
        url = "https://api.twitter.com/2/users/me"
        headers = {
            "Authorization": self._generate_oauth_header("GET", url)
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                user_data = response.json()
                return {
                    "success": True,
                    "user": user_data.get('data', {}),
                    "message": "Twitter API connection successful"
                }
            else:
                return {
                    "success": False,
                    "error": response.json() if response.text else f"HTTP {response.status_code}",
                    "message": "Twitter API connection failed"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to Twitter API"
            }

class VisibilityAutomator:
    """Enhanced content distribution with proper Twitter integration"""
    
    def __init__(self, config):
        self.config = config
        self.social_accounts = config.get('social_accounts', {})
        self.twitter_config = config.get('twitter_api', {})
        self.twitter_api = None
        
        # Initialize Twitter API with OAuth 1.0a (proper method)
        twitter_creds = self.twitter_config
        if all(key in twitter_creds for key in ['api_key', 'api_secret', 'access_token', 'access_token_secret']):
            self.twitter_api = TwitterAPI(
                api_key=twitter_creds['api_key'],
                api_secret=twitter_creds['api_secret'],
                access_token=twitter_creds['access_token'],
                access_token_secret=twitter_creds['access_token_secret']
            )
            print("Twitter API initialized with OAuth 1.0a ")
        else:
            print("Warning: Twitter OAuth credentials incomplete in config")
            missing = [key for key in ['api_key', 'api_secret', 'access_token', 'access_token_secret'] 
                      if key not in twitter_creds]
            print(f"Missing: {', '.join(missing)}")
        
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
    
    def post_to_twitter(self, post, custom_text: str = None) -> dict:
        """Post blog intro to Twitter"""
        if not self.twitter_api:
            return {
                "success": False,
                "error": "Twitter API not configured. Please check your OAuth credentials in config."
            }
        
        try:
            # Use custom text or generate social post
            if custom_text:
                tweet_text = custom_text
            else:
                social_posts = self.generate_social_posts(post)
                tweet_text = social_posts['twitter']
            
            print(f"Generated tweet text: {tweet_text}")
            print(f"Tweet length: {len(tweet_text)} characters")
            
            # Post the tweet
            result = self.twitter_api.post_tweet(tweet_text)
            
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
                    "text": tweet_text,
                    "status_code": result.get('status_code')
                }
        
        except Exception as e:
            self.logger.error(f"Twitter posting error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
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