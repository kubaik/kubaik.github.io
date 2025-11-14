import os
import json
import random
import re  
import yaml
import asyncio
import aiohttp
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from blog_post import BlogPost
from monetization_manager import MonetizationManager
from seo_optimizer import SEOOptimizer
from visibility_automator import VisibilityAutomator
from static_site_generator import StaticSiteGenerator
from hashtag_manager import HashtagManager, add_hashtags_to_post

#BlogSystem
class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Use Groq API instead of OpenAI
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize monetization manager
        self.monetization = MonetizationManager(config)
        
        # Initialize hashtag manager
        self.hashtag_manager = HashtagManager(config)

    def cleanup_posts(self):
        """Clean ups incomplete posts and recover from markdown files"""
        print("Cleaning up posts...")
        
        if not self.output_dir.exists():
            print("No docs directory found.")
            return
        
        fixed_count = 0
        removed_count = 0
        
        for post_dir in self.output_dir.iterdir():
            if not post_dir.is_dir():
                continue
            
            post_json_path = post_dir / "post.json"
            markdown_path = post_dir / "index.md"
            
            if not post_json_path.exists() and markdown_path.exists():
                try:
                    print(f"Recovering {post_dir.name}...")
                    post = BlogPost.from_markdown_file(markdown_path, post_dir.name)
                    self.save_post(post)
                    fixed_count += 1
                    print(f"Recovered: {post.title}")
                except Exception as e:
                    print(f"Failed to recover {post_dir.name}: {e}")
            
            elif not post_json_path.exists() and not markdown_path.exists():
                print(f"Removing empty directory: {post_dir.name}")
                try:
                    post_dir.rmdir()
                    removed_count += 1
                except OSError:
                    print(f"Directory not empty: {list(post_dir.iterdir())}")
        
        print(f"Cleanup complete: {fixed_count} recovered, {removed_count} removed")

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            print("No Groq API key found. Using fallback content generation.")
            return self._generate_fallback_post(topic)
        
        try:
            print(f"Generating content for: {topic}")
            print(f"Using Groq API (llama-3.3-70b-versatile)")
            
            title = await self._generate_title(topic, keywords)
            content = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(topic, title)
            slug = self._create_slug(title)
            
            if not keywords:
                keywords = await self._generate_keywords(topic, title)
            
            post = BlogPost(
                title=title.strip(),
                content=content.strip(),
                slug=slug,
                tags=keywords[:5],
                meta_description=meta_description.strip(),
                featured_image=f"/static/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=keywords,
                affiliate_links=[],
                monetization_data={}
            )
            
            # Process for monetization
            enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
                post.content, topic
            )
            post.content = enhanced_content
            post.affiliate_links = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)
            
            # Add daily trending hashtags
            print("✨ Generating trending hashtags...")
            hashtags = await self.hashtag_manager.get_daily_hashtags(topic, max_hashtags=10)
            post.tags = list(set(post.tags + hashtags))[:15]  # Merge and limit
            post.seo_keywords = list(set(post.seo_keywords + hashtags))[:15]  # Add to SEO keywords too
            
            # Store formatted hashtags for social media
            post.twitter_hashtags = self.hashtag_manager.format_hashtags_for_twitter(hashtags[:5])
            
            print(f"📱 Hashtags: {', '.join(hashtags[:5])}")
            
            return post
            
        except Exception as e:
            print(f"Error generating blog post: {e}")
            print("Falling back to sample content...")
            return self._generate_fallback_post(topic)

    def _generate_fallback_post(self, topic: str) -> BlogPost:
        """Generate a fallback post when API is unavailable"""
        title = f"Understanding {topic}: A Complete Guide"
        slug = self._create_slug(title)
        
        content = f"""## Introduction

{topic} is a crucial aspect of modern technology that every developer should understand. In this comprehensive guide, we'll explore the key concepts and best practices.

## What is {topic}?

{topic} represents an important area of technology development that has gained significant traction in recent years. Understanding its core principles is essential for building effective solutions.

## Key Benefits

- **Improved Performance**: {topic} can significantly enhance system performance
- **Better Scalability**: Implementing {topic} helps applications scale more effectively  
- **Enhanced User Experience**: Users benefit from the improvements that {topic} brings
- **Cost Effectiveness**: Proper implementation can reduce operational costs

## Best Practices

### 1. Planning and Strategy

Before implementing {topic}, it's important to have a clear strategy and understanding of your requirements.

### 2. Implementation Approach

Take a systematic approach to implementation, starting with the fundamentals and building up complexity gradually.

### 3. Testing and Optimization

Regular testing and optimization ensure that your {topic} implementation continues to perform well.

## Common Challenges

When working with {topic}, developers often encounter several common challenges:

1. **Complexity Management**: Keeping implementations simple and maintainable
2. **Performance Optimization**: Ensuring optimal performance across different scenarios
3. **Integration Issues**: Seamlessly integrating with existing systems

## Conclusion

{topic} is an essential technology for modern development. By following best practices and understanding the core concepts, you can successfully implement solutions that deliver real value.

Remember to stay updated with the latest developments in {topic} as the field continues to evolve rapidly."""

        post = BlogPost(
            title=title,
            content=content,
            slug=slug,
            tags=[topic.replace(' ', '-').lower(), 'technology', 'development', 'guide'],
            meta_description=f"A comprehensive guide to {topic} covering key concepts, benefits, and best practices for developers.",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            seo_keywords=[topic.lower(), 'guide', 'tutorial', 'best practices'],
            affiliate_links=[],
            monetization_data={}
        )
        
        # Process for monetization
        enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
            post.content, topic
        )
        post.content = enhanced_content
        post.affiliate_links = affiliate_links
        post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)
        
        return post

    async def _call_groq_api(self, messages: List[Dict], max_tokens: int = 1000):
        """Call Groq API (OpenAI-compatible endpoint)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",  # Groq's best model
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.groq.com/openai/v1/chat/completions", 
                                   headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Groq API error: {response.status} - {error_text}")

    async def _generate_title(self, topic: str, keywords: List[str] = None) -> str:
        keyword_text = f" Focus on keywords: {', '.join(keywords)}" if keywords else ""
        
        messages = [
            {"role": "system", "content": "You are a skilled blog title writer. Create engaging, SEO-friendly titles."},
            {"role": "user", "content": f"Generate a compelling blog post title about '{topic}'.{keyword_text} The title should be catchy, informative, and under 60 characters."}
        ]
        
        title = await self._call_groq_api(messages, max_tokens=100)
        return title.strip().strip('"')

    async def _generate_content(self, title: str, topic: str, keywords: List[str] = None) -> str:
        keyword_text = f"\nKeywords to incorporate naturally: {', '.join(keywords)}" if keywords else ""
        
        messages = [
            {
                "role": "system", 
                "content": """You are an experienced tech blogger who writes detailed, 
                practical articles. Always include specific examples, code snippets, 
                real numbers, and actionable insights. Avoid generic statements and filler text."""
            },
            {
                "role": "user", 
                "content": f"""Write a 1,500-word technical blog post with the title: "{title}"

Topic: {topic}{keyword_text}

Requirements:
- Write in Markdown format (##, ###)
- Include 2-3 practical code examples with explanations
- Mention specific tools, platforms, or services by name
- Include real metrics, pricing data, or performance benchmarks where relevant
- Provide concrete use cases with implementation details
- Address common problems with specific solutions
- Write 1,200-1,800 words of substantial content
- Use bullet points and numbered lists where appropriate
- Add a strong conclusion with actionable next steps

Avoid:
- Generic phrases like "crucial aspect", "important technology", or "plays a vital role"
- Vague benefits without specifics
- Template-like structure
- Filler content that doesn't add value
- Repeating the same points in different words

Do not include the main title (# {title}) as it will be added automatically."""
            }
        ]
        
        content = await self._call_groq_api(messages, max_tokens=3000)
        return content.strip()

    async def _generate_meta_description(self, topic: str, title: str) -> str:
        messages = [
            {"role": "system", "content": "You create SEO-optimized meta descriptions."},
            {"role": "user", "content": f"Write a compelling meta description (under 160 characters) for a blog post titled '{title}' about {topic}."}
        ]
        
        description = await self._call_groq_api(messages, max_tokens=100)
        return description.strip().strip('"')

    async def _generate_keywords(self, topic: str, title: str) -> List[str]:
        messages = [
            {"role": "system", "content": "You generate relevant SEO keywords."},
            {"role": "user", "content": f"Generate 8-10 relevant SEO keywords for a blog post titled '{title}' about {topic}. Return as a comma-separated list."}
        ]
        
        keywords_text = await self._call_groq_api(messages, max_tokens=150)
        keywords = [k.strip().strip('"') for k in keywords_text.split(',')]
        return [k for k in keywords if k][:10]

    def _create_slug(self, title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]

    def save_post(self, post: BlogPost):
        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)
        
        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post.to_dict(), f, indent=2, ensure_ascii=False)
        
        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")
        
        print(f"Saved post: {post.title} ({post.slug})")
        if post.affiliate_links:
            print(f"  • {len(post.affiliate_links)} affiliate links added")
        print(f"  • {post.monetization_data.get('ad_slots', 0)} ad slots configured")


def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    print(f"Picking topic from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please create it first.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    topics = config.get("content_topics", [])
    if not topics:
        raise ValueError("No content_topics found in config.yaml")
    
    used = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                used = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            used = []
    
    available = [t for t in topics if t not in used]
    if not available:
        print("All topics used, resetting...")
        available = topics
        used = []
    
    topic = random.choice(available)
    used.append(topic)
    
    with open(history_file, "w") as f:
        json.dump(used, f, indent=2)
    
    print(f"Selected topic: {topic}")
    return topic


def create_sample_config():
    """Create a sample config.yaml file with monetization settings"""
    config = {
        "site_name": "AI Tech Blog",
        "site_description": "Cutting-edge insights into technology, AI, and development",
        "base_url": "https://kubaik.github.io",
        "base_path": "",
        
        # Monetization settings
        "amazon_affiliate_tag": "aiblogcontent-20",
        "google_analytics_id": "G-DST4PJYK6V",
        "google_adsense_id": "ca-pub-4477679588953789",
        "google_search_console_key": "AIzaSyBqIII5-K2quNev9w7iJoH5U4uqIqKDkEQ",
        "google_adsense_verification":"ca-pub-4477679588953789",
        
        # Social media accounts
        "social_accounts": {
            "twitter": "@KubaiKevin",
            "linkedin": "your-linkedin-page",
            "facebook": "your-facebook-page"
        },

        "twitter_api":{
        "api_key": "xXYqgAm2Gu3sgAPhzcS8bHR3i",
        "api_secret": "FdHGXdNHIKRb5ixTwsyZ9ITc25pYTsOhn1Wq0jDdOzI18M7frm",
        "access_token": "263530787-mN4pGjKKmroFPzBSpVRXP7bb4euKMryGnH6o8gBC",
        "access_token_secret": "a8LwH56vRImFaOXrlCaw7lHD0yu5i1wVHJBIllnBJLB9V",
        "bearer_token": "AAAAAAAAAAAAAAAAAAAAAGAYpwEAAAAAgMWEPoCpeWXsTJ0BlJCpXWGP3qU%3DryCORdvSbcYa9pfDeajtl8zwcAYd86wYuDvAZxCYlhfQr49HBO"
        },
  
        
        "content_topics": [
            "Machine Learning Algorithms",
            "Web Development Trends", 
            "Data Science Techniques",
            "Artificial Intelligence Applications",
            "Cloud Computing Platforms",
            "Cybersecurity Best Practices",
            "Mobile App Development",
            "DevOps and CI/CD",
            "Database Optimization",
            "Frontend Frameworks",
            "Backend Architecture",
            "API Design Patterns",
            "Software Testing Strategies",
            "Performance Optimization",
            "Blockchain Technology",
            "Internet of Things (IoT)",
            "Microservices Architecture",
            "Container Technologies",
            "Serverless Computing",
            "Progressive Web Apps"
        ]
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("Created sample config.yaml file with monetization settings")
    print("\n📝 Next steps:")
    print("1. Replace 'your-tag-20' with your Amazon Associates tag")
    print("2. Add your Google Analytics 4 measurement ID")
    print("3. Add your Google AdSense ID (ca-pub-xxxxxxxxxx)")
    print("4. Update social media handles")
    print("5. Consider applying for affiliate programs like:")
    print("   • ShareASale")
    print("   • Commission Junction")
    print("   • DigitalOcean referral program")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "init":
            print("Initializing enhanced blog system...")
            create_sample_config()
            os.makedirs("docs/static", exist_ok=True)
            os.makedirs("analytics", exist_ok=True)
            print("Blog system initialized with monetization features!")
            print("\n🚀 Now using Groq API (Free & Fast!)")
            print("Add GROQ_API_KEY to your environment or .env file")
            
        elif mode == "auto":
            print("Starting automated blog generation with monetization...")
            print("🚀 Using Groq API (llama-3.3-70b-versatile)")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found. Run 'python blog_system.py init' first.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            
            try:
                topic = pick_next_topic()
                blog_post = asyncio.run(blog_system.generate_blog_post(topic))
                blog_system.save_post(blog_post)
                
                generator = StaticSiteGenerator(blog_system)
                generator.generate_site()
                
                print(f"✅ Enhanced post '{blog_post.title}' generated successfully!")

                visibility = VisibilityAutomator(config)
                
                # Create tweet with hashtags
                hashtags = blog_post.twitter_hashtags if hasattr(blog_post, 'twitter_hashtags') else ""
                tweet_text = f"🚀 New Post: {blog_post.title}\n\n{blog_post.meta_description[:100]}...\n\n🔗 Read more: https://kubaik.github.io/{blog_post.slug}\n\n{hashtags}"
                
                # Ensure tweet is under 280 characters
                if len(tweet_text) > 280:
                    tweet_text = f"🚀 {blog_post.title}\nRead: https://kubaik.github.io/{blog_post.slug}\n{hashtags}"
                
                twitter_result = visibility.post_to_twitter(tweet_text)
                if twitter_result['success']:
                    print(f"🐦 Tweeted successfully URL: {twitter_result['url']}")
                else:
                    print(f"⚠️ Twitter post failed: {twitter_result.get('error')}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
        elif mode == "build":
            print("Building static site with monetization features...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Enhanced site rebuilt successfully!")
            
        elif mode == "cleanup":
            print("Running cleanup with monetization enhancements...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            blog_system.cleanup_posts()
            
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("✅ Cleanup and enhanced rebuild complete!")
            
        elif mode == "debug":
            print("Debug mode with monetization analysis...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            
            print(f"Output directory: {blog_system.output_dir}")
            print(f"Directory exists: {blog_system.output_dir.exists()}")
            
            if blog_system.output_dir.exists():
                items = list(blog_system.output_dir.iterdir())
                print(f"Items in directory: {len(items)}")
                for item in items:
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        post_json = item / "post.json"
                        post_md = item / "index.md"
                        social_json = item / "social_posts.json"
                        print(f"    post.json: {'Yes' if post_json.exists() else 'No'}")
                        print(f"    index.md: {'Yes' if post_md.exists() else 'No'}")
                        print(f"    social_posts.json: {'Yes' if social_json.exists() else 'No'}")
                        if post_json.exists():
                            try:
                                with open(post_json, 'r') as f:
                                    data = json.load(f)
                                print(f"    Valid post: {data.get('title', 'Unknown')}")
                                print(f"    Affiliate links: {len(data.get('affiliate_links', []))}")
                                print(f"    Ad slots: {data.get('monetization_data', {}).get('ad_slots', 0)}")
                            except Exception as e:
                                print(f"    Invalid JSON: {e}")
            
            print("\nRunning automatic cleanup with monetization...")
            blog_system.cleanup_posts()
            
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            
        elif mode == "social":
            print("Generating social media posts for existing content...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            posts = generator._get_all_posts()
            
            visibility = VisibilityAutomator(config)
            
            print(f"Generating social media posts for {len(posts)} posts...")
            for post in posts:
                social_posts = visibility.generate_social_posts(post)
                
                post_dir = blog_system.output_dir / post.slug
                social_file = post_dir / "social_posts.json"
                with open(social_file, 'w', encoding='utf-8') as f:
                    json.dump(social_posts, f, indent=2)
                
                print(f"Social posts generated for: {post.title}")
                print(f"  Twitter: {social_posts['twitter'][:50]}...")
                print(f"  LinkedIn: {social_posts['linkedin'][:50]}...")
                print(f"  Reddit: {social_posts['reddit_title']}")
                print()
            
            print("Social media posts generated for all posts!")
        
        elif mode == "test-twitter":
            print("Testing Twitter integration...")
            
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            visibility = VisibilityAutomator(config)
            
            # Test connection
            connection_test = visibility.test_twitter_connection()
            print(f"Connection test: {connection_test}")
            
            if connection_test['success']:
                # Create a simple test post
                class TestPost:
                    def __init__(self):
                        self.title = "Test - AI Blog System Twitter Integration"
                        self.meta_description = "Testing our automated blog to Twitter posting system. Integration test successful!"
                        self.slug = "test-twitter-integration"
                        self.tags = ["test", "automation", "blogging"]
                
                test_post = TestPost()
                
                # Generate social post
                social_posts = visibility.generate_social_posts(test_post)
                print(f"\nGenerated Twitter post preview:")
                print(f"📱 {social_posts['twitter']}")
                print(f"   Length: {len(social_posts['twitter'])} characters")
                
                # Ask for confirmation before posting
                response = input("\nPost this test tweet? (y/N): ")
                if response.lower() == 'y':
                    result = visibility.post_to_twitter(test_post)
                    if result['success']:
                        print("✅ Test tweet posted successfully!")
                        print(f"🆔 Tweet ID: {result.get('tweet_id')}")
                    else:
                        print(f"❌ Test tweet failed: {result['error']}")
                else:
                    print("Test cancelled - no tweet posted.")   

        else:
            print("Usage: python blog_system.py [init|auto|build|cleanup|debug|social|test-twitter]")
            print("  init         - Initialize blog system with monetization config")
            print("  auto         - Generate new post with monetization and rebuild site")
            print("  build        - Rebuild site with monetization features")
            print("  cleanup      - Fix missing files and rebuild with monetization")
            print("  debug        - Debug current state with monetization analysis")
            print("  social       - Generate social media posts for existing content")
            print("  test-twitter - Test Twitter API integration")
    else:
        print("Enhanced AI Blog System with Monetization & AdSense")
        print("🚀 Now powered by Groq API (FREE & FAST!)")
        print("\nUsage: python blog_system.py [command]")
        print("\nAvailable commands:")
        print("  init         - Initialize blog system with monetization settings")
        print("  auto         - Generate new monetized post and rebuild site")
        print("  build        - Rebuild site with all monetization features")
        print("  cleanup      - Fix posts and rebuild with enhancements")
        print("  debug        - Analyze current state and rebuild")
        print("  social       - Generate social media posts for promotion")
        print("  test-twitter - Test Twitter API connection and posting")
        print("\nMonetization features included:")
        print("  • Automated affiliate link injection")
        print("  • Google AdSense integration with responsive ads")
        print("  • Strategic ad placement slots (header, middle, footer)")
        print("  • SEO optimization with structured data")
        print("  • Social media post generation")
        print("  • RSS feed for subscribers (/rss.xml)")
        print("  • Search engine submission")
        print("  • Revenue estimation and reporting")
        print("\nSetup required:")
        print("  1. Run 'init' to create config.yaml")
        print("  2. Add GROQ_API_KEY to environment/GitHub secrets")
        print("  3. Add your Google AdSense ID (ca-pub-xxxxxxxxxx)")
        print("  4. Add your Google Analytics ID")
        print("  5. Configure affiliate program IDs")