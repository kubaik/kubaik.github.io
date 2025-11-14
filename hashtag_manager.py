"""
Intelligent Hashtag Manager - No External API Required
Uses Groq AI to generate daily trending tech hashtags
"""

import os
import json
import aiohttp
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set


class HashtagManager:
    """Manages trending and static hashtags for blog posts using Groq AI"""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache_file = Path(".hashtag_cache.json")
        self.cache_duration = timedelta(hours=12)  # Refresh twice daily
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Always-relevant core tech hashtags
        self.core_tech_hashtags = [
            "tech", "technology", "AI", "programming", "coding",
            "developer", "software", "innovation", "techtrends",
            "MachineLearning", "WebDev", "Cloud", "DevOps",
            "Cybersecurity", "DataScience", "IoT", "Blockchain"
        ]
        
        # Topic-specific hashtag mapping
        self.topic_hashtags = {
            "machine learning": ["ML", "AI", "DeepLearning", "NeuralNetworks", "DataScience", "TensorFlow", "PyTorch"],
            "web development": ["WebDev", "JavaScript", "React", "Frontend", "Backend", "FullStack", "WebDesign"],
            "data science": ["DataScience", "BigData", "Analytics", "Python", "ML", "Statistics", "DataViz"],
            "artificial intelligence": ["AI", "MachineLearning", "DeepLearning", "NeuralNetworks", "AGI", "AIResearch"],
            "cloud computing": ["Cloud", "AWS", "Azure", "GCP", "CloudComputing", "CloudNative", "Serverless"],
            "cybersecurity": ["Cybersecurity", "InfoSec", "Security", "CyberSec", "Privacy", "Encryption", "ZeroTrust"],
            "mobile app": ["MobileApp", "iOS", "Android", "AppDev", "MobileDev", "ReactNative", "Flutter"],
            "devops": ["DevOps", "CI", "CD", "Docker", "Kubernetes", "GitOps", "Automation"],
            "database": ["Database", "SQL", "NoSQL", "DataManagement", "PostgreSQL", "MongoDB", "Redis"],
            "frontend": ["Frontend", "React", "Vue", "Angular", "JavaScript", "CSS", "HTML5"],
            "backend": ["Backend", "API", "NodeJS", "Python", "Java", "ServerSide", "Microservices"],
            "api": ["API", "REST", "GraphQL", "WebServices", "APIDesign", "OpenAPI", "APIFirst"],
            "testing": ["Testing", "QA", "Automation", "TestDriven", "UnitTesting", "Selenium", "Jest"],
            "performance": ["Performance", "Optimization", "Speed", "Efficiency", "LoadTime", "WebPerf"],
            "blockchain": ["Blockchain", "Crypto", "Web3", "SmartContracts", "DeFi", "NFT", "Ethereum"],
            "iot": ["IoT", "InternetOfThings", "SmartDevices", "EdgeComputing", "IIoT", "Sensors"],
            "microservices": ["Microservices", "Architecture", "Cloud", "Containers", "ServiceMesh", "API"],
            "container": ["Docker", "Kubernetes", "Containers", "DevOps", "CloudNative", "Podman"],
            "serverless": ["Serverless", "Lambda", "FaaS", "Cloud", "Functions", "CloudFunctions"],
            "progressive web": ["PWA", "WebDev", "MobileWeb", "Progressive", "ServiceWorker", "OfflineFirst"]
        }
        
        # Curated trending tech hashtags (manually updated periodically - 2024/2025)
        self.trending_tech_hashtags = [
            # AI & ML Hot Topics
            "AI2024", "GenerativeAI", "ChatGPT", "LLM", "OpenAI", "Claude", "Gemini",
            "AITools", "PromptEngineering", "LangChain", "VectorDB",
            
            # Development Communities
            "100DaysOfCode", "CodeNewbie", "DevCommunity", "WomenWhoCode", "TechTwitter",
            "BuildInPublic", "IndieDev", "IndieHackers",
            
            # Hot Technologies 2024/2025
            "Python", "JavaScript", "TypeScript", "Rust", "Go", "Swift", "Kotlin",
            "React", "NextJS", "Vue", "Svelte", "TailwindCSS", "Astro",
            
            # Platforms & Tools
            "GitHub", "GitLab", "VSCode", "Docker", "Kubernetes", "Vercel", "Supabase",
            
            # Trending Concepts
            "CleanCode", "BestPractices", "TechTips", "LearnToCode", "CodeReview",
            "OpenSource", "TechNews", "StartupLife", "RemoteWork", "DigitalNomad",
            
            # Emerging Tech
            "EdgeComputing", "QuantumComputing", "5G", "AR", "VR", "Metaverse",
            "GreenTech", "CleanEnergy", "SustainableTech"
        ]
    
    async def get_daily_hashtags(self, topic: str, max_hashtags: int = 10) -> List[str]:
        """
        Get a mix of AI-generated, trending, and relevant hashtags for the topic
        Returns: List of hashtags without the # symbol
        """
        hashtags = set()
        
        # 1. Add AI-generated trending hashtags (3-4)
        ai_hashtags = await self._get_ai_generated_hashtags(topic)
        if ai_hashtags:
            hashtags.update(ai_hashtags[:4])
        
        # 2. Add topic-specific hashtags (2-3)
        topic_specific = self._get_topic_specific_hashtags(topic)
        hashtags.update(topic_specific[:3])
        
        # 3. Add core tech hashtags (1-2)
        hashtags.update(random.sample(self.core_tech_hashtags, min(2, len(self.core_tech_hashtags))))
        
        # 4. Add curated trending hashtags (1-2)
        hashtags.update(random.sample(self.trending_tech_hashtags, min(2, len(self.trending_tech_hashtags))))
        
        # 5. Ensure we have the right number
        hashtags_list = list(hashtags)
        random.shuffle(hashtags_list)
        
        return hashtags_list[:max_hashtags]
    
    def _get_topic_specific_hashtags(self, topic: str) -> List[str]:
        """Get hashtags specific to the topic"""
        topic_lower = topic.lower()
        
        # Find matching topic category
        for key, tags in self.topic_hashtags.items():
            if key in topic_lower:
                selected = random.sample(tags, min(3, len(tags)))
                return selected
        
        # If no match, return general tech tags
        return random.sample(self.core_tech_hashtags, 3)
    
    async def _get_ai_generated_hashtags(self, topic: str) -> List[str]:
        """
        Use Groq AI to generate trending, relevant hashtags for the topic
        This is the smart part - AI understands current trends!
        """
        # Check cache first
        cached = self._load_cache(topic)
        if cached:
            return cached
        
        if not self.groq_api_key:
            print("⚠️ No Groq API key - using curated hashtags only")
            return []
        
        try:
            print(f"🤖 Generating AI hashtags for: {topic}")
            
            prompt = f"""Generate 5-7 trending, relevant hashtags for a tech blog post about "{topic}".

Requirements:
- Return ONLY hashtags, one per line
- No # symbol (just the text)
- Mix of popular and specific hashtags
- Focus on 2024/2025 trending tech topics
- Make them Twitter/X friendly (short, memorable)
- Consider current AI, development, and tech trends

Example format:
AI
MachineLearning
TechTrends
Python
DataScience

Now generate hashtags for: {topic}"""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a social media expert who generates trending hashtags. Return ONLY hashtags, one per line, no # symbol, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.8  # Higher temperature for creative hashtags
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # Parse hashtags from response
                        hashtags = []
                        for line in content.strip().split('\n'):
                            # Clean up the line
                            tag = line.strip().strip('#').strip()
                            
                            # Skip empty lines and explanations
                            if tag and not any(c in tag for c in [',', '.', ':', ' ']):
                                hashtags.append(tag)
                        
                        # Cache the results
                        if hashtags:
                            self._save_cache(topic, hashtags)
                            print(f"✅ Generated {len(hashtags)} AI hashtags")
                            return hashtags[:7]
                    else:
                        print(f"⚠️ Groq API error: {response.status}")
        
        except Exception as e:
            print(f"⚠️ Error generating AI hashtags: {e}")
        
        return []
    
    def _load_cache(self, topic: str) -> List[str]:
        """Load hashtags from cache if not expired"""
        if not self.cache_file.exists():
            return []
        
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            # Create a cache key from topic
            cache_key = topic.lower().replace(' ', '_')
            
            if cache_key in cache:
                entry = cache[cache_key]
                cached_time = datetime.fromisoformat(entry['timestamp'])
                
                if datetime.now() - cached_time < self.cache_duration:
                    print(f"📦 Using cached hashtags for: {topic}")
                    return entry['hashtags']
        
        except Exception as e:
            print(f"⚠️ Error loading hashtag cache: {e}")
        
        return []
    
    def _save_cache(self, topic: str, hashtags: List[str]):
        """Save hashtags to cache"""
        try:
            # Load existing cache
            cache = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
            
            # Add new entry
            cache_key = topic.lower().replace(' ', '_')
            cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'hashtags': hashtags,
                'topic': topic
            }
            
            # Save back to file
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            print(f"💾 Cached hashtags for: {topic}")
        
        except Exception as e:
            print(f"⚠️ Error saving hashtag cache: {e}")
    
    def format_hashtags(self, hashtags: List[str], include_hash: bool = True) -> str:
        """Format hashtags for different platforms"""
        prefix = "#" if include_hash else ""
        return " ".join(f"{prefix}{tag}" for tag in hashtags)
    
    def format_hashtags_for_twitter(self, hashtags: List[str], max_length: int = 280) -> str:
        """Format hashtags for Twitter with character limit consideration"""
        formatted = []
        total_length = 0
        
        for tag in hashtags:
            tag_with_hash = f"#{tag}"
            tag_length = len(tag_with_hash) + 1  # +1 for space
            
            if total_length + tag_length <= max_length:
                formatted.append(tag_with_hash)
                total_length += tag_length
            else:
                break
        
        return " ".join(formatted)
    
    def get_hashtags_for_social_media(self, hashtags: List[str], platform: str = "twitter") -> str:
        """Get formatted hashtags for specific platform"""
        if platform.lower() == "twitter":
            return self.format_hashtags_for_twitter(hashtags[:5])  # Twitter best practice: 1-5 hashtags
        elif platform.lower() == "instagram":
            return self.format_hashtags(hashtags[:15])  # Instagram allows up to 30
        elif platform.lower() == "linkedin":
            return self.format_hashtags(hashtags[:5])  # LinkedIn best practice: 3-5 hashtags
        else:
            return self.format_hashtags(hashtags[:8])


# Integration function for blog_system.py
async def add_hashtags_to_post(post, config):
    """
    Helper function to add hashtags to a blog post
    Call this in blog_system.py after generating the post
    """
    manager = HashtagManager(config)
    
    # Get hashtags for the post topic
    hashtags = await manager.get_daily_hashtags(post.title, max_hashtags=10)
    
    # Add to post tags (merge with existing)
    existing_tags = set(post.tags) if post.tags else set()
    existing_tags.update(hashtags)
    post.tags = list(existing_tags)[:15]  # Limit to 15 total tags
    
    # Store formatted versions for social media
    post.twitter_hashtags = manager.format_hashtags_for_twitter(hashtags[:5])
    post.instagram_hashtags = manager.format_hashtags(hashtags[:15])
    post.linkedin_hashtags = manager.format_hashtags(hashtags[:5])
    
    print(f"✨ Added {len(hashtags)} trending hashtags to post")
    print(f"📱 Twitter hashtags: {post.twitter_hashtags}")
    
    return post


# Test function
async def test_hashtag_generation():
    """Test the hashtag generation system"""
    print("🧪 Testing Hashtag Generation System\n")
    
    config = {
        "site_name": "Test Blog"
    }
    
    manager = HashtagManager(config)
    
    # Test topics
    test_topics = [
        "Machine Learning Algorithms",
        "Web Development Trends",
        "Cybersecurity Best Practices",
        "Cloud Computing"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*70}")
        print(f"Topic: {topic}")
        print('='*70)
        
        hashtags = await manager.get_daily_hashtags(topic, max_hashtags=10)
        
        print(f"\n✅ Generated {len(hashtags)} hashtags:")
        print(f"   {', '.join(hashtags)}")
        
        print(f"\n📱 Formatted for platforms:")
        print(f"   Twitter: {manager.format_hashtags_for_twitter(hashtags)}")
        print(f"   Instagram: {manager.format_hashtags(hashtags[:15])}")
        print(f"   LinkedIn: {manager.format_hashtags(hashtags[:5])}")


if __name__ == "__main__":
    import asyncio
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              Intelligent Hashtag Manager - No API Required!               ║
║                                                                            ║
║  • Uses Groq AI to generate trending hashtags                            ║
║  • No external APIs needed                                                ║
║  • Caches results for 12 hours                                           ║
║  • Smart topic-specific selection                                         ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(test_hashtag_generation())