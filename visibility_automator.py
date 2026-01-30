"""
Enhanced Visibility Automator with Multiple Tweet Strategies
Integrates with EnhancedTweetGenerator for maximum engagement
"""

import tweepy
from typing import Dict
from enhanced_tweet_generator import EnhancedTweetGenerator


class VisibilityAutomator:
    def __init__(self, config):
        self.config = config
        self.twitter_client = None
        self.tweet_generator = EnhancedTweetGenerator()
        self._init_twitter()
    
    def _init_twitter(self):
        """Initialize Twitter API v2 client"""
        twitter_config = self.config.get('twitter_api', {})
        
        if not twitter_config:
            print("⚠️ No Twitter API credentials found in config")
            return
        
        try:
            self.twitter_client = tweepy.Client(
                bearer_token=twitter_config.get('bearer_token'),
                consumer_key=twitter_config.get('api_key'),
                consumer_secret=twitter_config.get('api_secret'),
                access_token=twitter_config.get('access_token'),
                access_token_secret=twitter_config.get('access_token_secret')
            )
            print("✅ Twitter API initialized")
        except Exception as e:
            print(f"❌ Twitter initialization failed: {e}")
            self.twitter_client = None
    
    def post_to_twitter(self, tweet_text: str = None, post = None, strategy: str = "auto") -> Dict:
        """
        Post to Twitter with enhanced engagement strategies
        
        Args:
            tweet_text: Custom tweet text (if provided, ignores post and strategy)
            post: Blog post object to generate tweet from
            strategy: Tweet strategy (hook, stat, problem, list, question, thread, auto)
        
        Returns:
            Dict with success status and tweet URL or error
        """
        if not self.twitter_client:
            return {
                'success': False,
                'error': 'Twitter client not initialized. Check API credentials.'
            }
        
        try:
            # Use custom text or generate engaging tweet
            if tweet_text:
                final_tweet = tweet_text
            elif post:
                final_tweet = self.tweet_generator.create_engaging_tweet(post, strategy)
            else:
                return {
                    'success': False,
                    'error': 'Either tweet_text or post must be provided'
                }
            
            # Analyze tweet quality before posting
            analysis = self.tweet_generator.analyze_tweet_quality(final_tweet)
            print(f"📊 Tweet Quality Score: {analysis['score']}/100 (Grade: {analysis['grade']})")
            
            # Post tweet
            response = self.twitter_client.create_tweet(text=final_tweet)
            tweet_id = response.data['id']
            
            # Get authenticated user info for URL
            me = self.twitter_client.get_me()
            username = me.data.username
            
            tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
            
            return {
                'success': True,
                'tweet_id': tweet_id,
                'url': tweet_url,
                'tweet_text': final_tweet,
                'quality_score': analysis['score'],
                'strategy': strategy
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def post_with_best_strategy(self, post) -> Dict:
        """
        Generate multiple variations and post the best one
        """
        if not self.twitter_client:
            return {
                'success': False,
                'error': 'Twitter client not initialized'
            }
        
        # Generate variations
        variations = self.tweet_generator.create_multiple_variations(post, count=5)
        
        # Analyze and pick best
        best_variation = max(variations, 
                           key=lambda v: self.tweet_generator.analyze_tweet_quality(v['tweet'])['score'])
        
        print(f"🎯 Selected strategy: {best_variation['strategy']}")
        print(f"📊 Quality score: {self.tweet_generator.analyze_tweet_quality(best_variation['tweet'])['score']}")
        
        # Post the best one
        return self.post_to_twitter(tweet_text=best_variation['tweet'], strategy=best_variation['strategy'])
    
    def generate_tweet_preview(self, post, strategy: str = "auto") -> Dict:
        """
        Generate tweet preview without posting
        Useful for review before posting
        """
        tweet = self.tweet_generator.create_engaging_tweet(post, strategy)
        analysis = self.tweet_generator.analyze_tweet_quality(tweet)
        
        return {
            'tweet': tweet,
            'length': len(tweet),
            'strategy': strategy,
            'quality_score': analysis['score'],
            'grade': analysis['grade'],
            'feedback': analysis['feedback']
        }
    
    def generate_all_variations(self, post) -> list:
        """
        Generate all tweet variations for manual selection
        """
        variations = self.tweet_generator.create_multiple_variations(post, count=6)
        
        results = []
        for var in variations:
            analysis = self.tweet_generator.analyze_tweet_quality(var['tweet'])
            results.append({
                'strategy': var['strategy'],
                'tweet': var['tweet'],
                'length': var['length'],
                'quality_score': analysis['score'],
                'grade': analysis['grade'],
                'feedback': analysis['feedback']
            })
        
        return sorted(results, key=lambda x: x['quality_score'], reverse=True)
    
    def test_twitter_connection(self) -> Dict:
        """Test Twitter API connection"""
        if not self.twitter_client:
            return {
                'success': False,
                'error': 'Twitter client not initialized'
            }
        
        try:
            me = self.twitter_client.get_me()
            return {
                'success': True,
                'username': me.data.username,
                'name': me.data.name,
                'id': me.data.id
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_social_posts(self, post) -> Dict:
        """
        Generate optimized social media posts for multiple platforms
        """
        # Twitter - use enhanced generator
        twitter_post = self.tweet_generator.create_engaging_tweet(post, strategy="auto")
        
        # LinkedIn - more professional
        linkedin_post = f"""🚀 New Article: {post.title}

{post.meta_description}

In this comprehensive guide, I cover:
✅ Key concepts and fundamentals
✅ Best practices from industry leaders
✅ Real-world implementation examples
✅ Common pitfalls to avoid

Perfect for developers and tech professionals looking to level up their skills.

Read the full article: https://kubaik.github.io/{post.slug}

{self._format_hashtags_linkedin(post)}
"""
        
        # Reddit - more casual, value-focused
        reddit_title = f"[Guide] {post.title}"
        reddit_post = f"""{post.meta_description}

I just published a detailed guide covering everything you need to know about this topic. 

The article includes:
- Step-by-step explanations
- Code examples and snippets  
- Performance benchmarks
- Best practices

Check it out if you're interested: https://kubaik.github.io/{post.slug}

Happy to answer any questions!
"""
        
        # Facebook - more casual and personal
        facebook_post = f"""Hey everyone! 👋

Just published a new article on {post.title}.

{post.meta_description}

If you've been curious about this topic or looking to improve your skills, this guide has you covered.

Read it here: https://kubaik.github.io/{post.slug}

Let me know what you think! 💬
"""
        
        return {
            'twitter': twitter_post,
            'linkedin': linkedin_post,
            'reddit_title': reddit_title,
            'reddit': reddit_post,
            'facebook': facebook_post
        }
    
    def _format_hashtags_linkedin(self, post) -> str:
        """Format hashtags for LinkedIn (different style)"""
        if hasattr(post, 'tags') and post.tags:
            tags = post.tags[:5]
            hashtags = []
            for tag in tags:
                clean_tag = tag.replace(' ', '').replace('-', '')
                if clean_tag:
                    hashtags.append(f"#{clean_tag}")
            return ' '.join(hashtags)
        return ""


# CLI for testing
if __name__ == "__main__":
    import yaml
    import sys
    
    print("=" * 70)
    print("ENHANCED TWEET GENERATOR - TESTING")
    print("=" * 70)
    
    # Mock post for testing
    class MockPost:
        def __init__(self):
            self.title = "Secure APIs: Best Practices for Protection"
            self.slug = "secure-apis"
            self.meta_description = "Learn API security best practices to protect your data and prevent breaches."
            self.tags = ["coding", "innovation", "CloudNative", "OpenAPI", "5G"]
            self.twitter_hashtags = "#coding #innovation #CloudNative"
    
    post = MockPost()
    
    # Load config if available
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    visibility = VisibilityAutomator(config)
    
    # Generate all variations
    print("\n🎨 Generating all tweet variations...")
    print("=" * 70)
    
    variations = visibility.generate_all_variations(post)
    
    for i, var in enumerate(variations, 1):
        print(f"\n📱 VARIATION {i}: {var['strategy'].upper()}")
        print(f"   Score: {var['quality_score']}/100 (Grade: {var['grade']})")
        print(f"   Length: {var['length']} characters")
        print("-" * 70)
        print(var['tweet'])
        print("-" * 70)
        print("Feedback:")
        for feedback in var['feedback']:
            print(f"   {feedback}")
    
    # Show best variation
    best = variations[0]
    print("\n" + "=" * 70)
    print("🏆 RECOMMENDED TWEET (Highest Score)")
    print("=" * 70)
    print(f"Strategy: {best['strategy'].upper()}")
    print(f"Score: {best['quality_score']}/100 (Grade: {best['grade']})")
    print(f"Length: {best['length']} characters")
    print("-" * 70)
    print(best['tweet'])
    print("=" * 70)
    
    # Test connection if credentials available
    if config.get('twitter_api'):
        print("\n🔍 Testing Twitter connection...")
        connection = visibility.test_twitter_connection()
        if connection['success']:
            print(f"✅ Connected as @{connection['username']}")
            
            response = input("\n❓ Do you want to post the best tweet? (y/N): ")
            if response.lower() == 'y':
                result = visibility.post_to_twitter(tweet_text=best['tweet'])
                if result['success']:
                    print(f"\n✅ Tweet posted successfully!")
                    print(f"🔗 URL: {result['url']}")
                else:
                    print(f"\n❌ Failed to post: {result['error']}")
        else:
            print(f"❌ Connection failed: {connection['error']}")
    else:
        print("\n⚠️ No Twitter credentials in config.yaml")
        print("Add your Twitter API credentials to test posting.")
