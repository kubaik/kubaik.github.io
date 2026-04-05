"""
Enhanced Twitter Post Generator for Maximum Engagement
Creates compelling, click-worthy tweets that drive traffic to blog posts
AI-friendly: No personal pronouns (I, my, etc.)
"""

import random
from typing import Dict, List

class EnhancedTweetGenerator:
    """Generate engaging tweets optimized for clicks and engagement"""
    
    # Attention-grabbing opening hooks (no personal pronouns)
    HOOKS = [
        "🚨 This changes everything:",
        "💡 Most developers miss this:",
        "⚡ Quick tip that saves hours:",
        "🔥 Hot take:",
        "🎯 Pro tip:",
        "📊 New data shows:",
        "⚠️ Common mistake to avoid:",
        "✨ Game-changer:",
        "🧵 Thread on why this matters:",
        "💰 This saves thousands:",
        "🤯 Mind-blowing fact:",
        "📚 Essential knowledge:",
        "🔍 Deep dive into:",
        "⏰ Just published:",
        "🚀 New guide:",
    ]
    
    # Value propositions (neutral voice)
    VALUE_PROPS = [
        "Learn how to {action} in under {time}",
        "Discover {number} ways to improve {topic}",
        "Master {skill} with this guide",
        "Avoid these {number} costly mistakes",
        "Get {benefit} without {pain_point}",
        "The complete guide to {topic}",
        "{number} secrets that {benefit}",
        "Why {claim} and what to do about it",
        "How to {action} like a pro",
        "The ultimate {topic} breakdown",
    ]
    
    # Call-to-action phrases
    CTAS = [
        "Read the full breakdown 👇",
        "See the complete guide 👇",
        "Get all the details here 👇",
        "Learn the full strategy 👇",
        "Check out the full tutorial 👇",
        "Dive deeper here 👇",
        "Full thread in the article 👇",
        "Everything you need to know 👇",
        "Step-by-step guide here 👇",
        "Read more 👇",
    ]
    
    # Curiosity gaps
    CURIOSITY = [
        "The results will surprise you.",
        "Number 3 is counterintuitive.",
        "This isn't what you'd expect.",
        "The last one is a game-changer.",
        "Most people get this wrong.",
        "The data is shocking.",
        "This challenges conventional wisdom.",
        "Wait until you see the benchmarks.",
        "The twist? It's actually simple.",
        "Plot twist at the end.",
    ]
    
    @staticmethod
    def create_engaging_tweet(post, strategy: str = "auto") -> str:
        """
        Create an engaging tweet using different strategies
        
        Strategies:
        - hook: Start with attention-grabbing hook
        - stat: Lead with compelling statistic or fact
        - problem: Present a problem and solution
        - list: Use numbered list format
        - question: Start with engaging question
        - thread: Thread-style preview
        - auto: Automatically choose best strategy
        """
        
        if strategy == "auto":
            strategy = random.choice(["hook", "stat", "problem", "list", "question", "thread"])
        
        if strategy == "hook":
            return EnhancedTweetGenerator._hook_tweet(post)
        elif strategy == "stat":
            return EnhancedTweetGenerator._stat_tweet(post)
        elif strategy == "problem":
            return EnhancedTweetGenerator._problem_tweet(post)
        elif strategy == "list":
            return EnhancedTweetGenerator._list_tweet(post)
        elif strategy == "question":
            return EnhancedTweetGenerator._question_tweet(post)
        elif strategy == "thread":
            return EnhancedTweetGenerator._thread_tweet(post)
        else:
            return EnhancedTweetGenerator._hook_tweet(post)
    
    @staticmethod
    def _hook_tweet(post) -> str:
        """Tweet starting with attention-grabbing hook"""
        hook = random.choice(EnhancedTweetGenerator.HOOKS)
        cta = random.choice(EnhancedTweetGenerator.CTAS)
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        
        # Extract key benefit from title or description
        title = post.title.replace(":", " -")
        
        tweet = f"{hook}\n\n{title}\n\n{cta}\n🔗 https://kubaik.github.io/{post.slug}\n\n{hashtags}"
        
        return EnhancedTweetGenerator._trim_tweet(tweet)
    
    @staticmethod
    def _stat_tweet(post) -> str:
        """Tweet leading with statistic or compelling fact"""
        stats = [
            "📊 85% of developers struggle with this.",
            "⚡ This technique improved performance by 300%.",
            "💰 Companies save $10K/month using this approach.",
            "⏰ Cut development time in half with these tips.",
            "🎯 98% accuracy using this method.",
            "📈 Traffic increased by 500% after implementing this.",
            "🔥 Over 10,000 developers already using this.",
        ]
        
        stat = random.choice(stats)
        title = post.title.replace(":", " -")
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        
        tweet = f"{stat}\n\n{title}\n\nFull guide here 👇\n🔗 https://kubaik.github.io/{post.slug}\n\n{hashtags}"
        
        return EnhancedTweetGenerator._trim_tweet(tweet)
    
    @staticmethod
    def _problem_tweet(post) -> str:
        """Tweet presenting a problem and teasing the solution"""
        # Extract topic from title
        topic = post.title.split(":")[0] if ":" in post.title else post.title
        
        tweet = f"Struggling with {topic}?\n\nThis complete guide covers:\n"
        tweet += f"✅ Best practices\n"
        tweet += f"✅ Common pitfalls\n"
        tweet += f"✅ Real examples\n\n"
        tweet += f"Read here 👇\n🔗 https://kubaik.github.io/{post.slug}\n\n"
        tweet += EnhancedTweetGenerator._format_hashtags(post)
        
        return EnhancedTweetGenerator._trim_tweet(tweet)
    
    @staticmethod
    def _list_tweet(post) -> str:
        """Tweet using numbered list format"""
        numbers = ["5", "7", "10"]
        num = random.choice(numbers)
        
        list_formats = [
            f"{num} things you need to know about",
            f"{num} ways to improve your",
            f"{num} mistakes to avoid when",
            f"{num} secrets to mastering",
            f"{num} best practices for",
        ]
        
        format_text = random.choice(list_formats)
        topic = post.title.split(":")[0] if ":" in post.title else post.title
        
        tweet = f"{format_text} {topic}:\n\n"
        tweet += f"1. [Preview in article]\n"
        tweet += f"2. [Preview in article]\n"
        tweet += f"3. [Preview in article]\n"
        tweet += f"...\n\n"
        tweet += f"Full list with examples 👇\n🔗 https://kubaik.github.io/{post.slug}\n\n"
        tweet += EnhancedTweetGenerator._format_hashtags(post)
        
        return EnhancedTweetGenerator._trim_tweet(tweet)
    
    @staticmethod
    def _question_tweet(post) -> str:
        """Tweet starting with engaging hook (varied openings)"""
        topic = post.title.split(":")[0] if ":" in post.title else post.title
        
        # Diverse hook variations - prevents repetition!
        hook_variations = [
            f"🔥 {topic} just got easier",
            f"⚡ Stop making these {topic} mistakes",
            f"💡 What nobody tells you about {topic}",
            f"🚀 Level up your {topic} game",
            f"🎯 The one thing about {topic} that matters",
            f"📊 {topic} in 2026: What changed?",
            f"⭐ {topic} secrets nobody shares",
            f"🔓 Unlock {topic} mastery",
            f"💪 {topic}: From beginner to pro",
            f"🌟 Why {topic} matters more than ever",
            f"⚠️ Common {topic} pitfalls to avoid",
            f"✨ {topic} made simple",
            f"🎓 Everything about {topic}",
            f"💎 {topic} best practices revealed",
            f"🧠 {topic} explained (finally)",
        ]
        
        question = random.choice(hook_variations)
        
        tweet = f"{question}\n\n"
        tweet += f"New comprehensive guide covering:\n\n"
        tweet += f"✨ Core concepts\n"
        tweet += f"🔧 Practical examples\n"
        tweet += f"⚡ Performance tips\n"
        tweet += f"🎯 Best practices\n\n"
        tweet += f"Dive in 👇\n🔗 https://kubaik.github.io/{post.slug}\n\n"
        tweet += EnhancedTweetGenerator._format_hashtags(post)
        
        return EnhancedTweetGenerator._trim_tweet(tweet)
    
    @staticmethod
    def _thread_tweet(post) -> str:
        """Tweet teasing a thread-style deep dive"""
        tweet = f"🧵 THREAD: Everything you need to know about {post.title}\n\n"
        tweet += f"Complete breakdown with:\n"
        tweet += f"→ Step-by-step guide\n"
        tweet += f"→ Real-world examples\n"
        tweet += f"→ Code snippets\n"
        tweet += f"→ Pro tips\n\n"
        tweet += f"Full article 👇\n🔗 https://kubaik.github.io/{post.slug}\n\n"
        tweet += EnhancedTweetGenerator._format_hashtags(post)
        
        return EnhancedTweetGenerator._trim_tweet(tweet)
    
    @staticmethod
    def _format_hashtags(post) -> str:
        """Format hashtags from post tags"""
        if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags:
            return post.twitter_hashtags
        
        if hasattr(post, 'tags') and post.tags:
            # Take first 3-5 tags and format as hashtags
            tags = post.tags[:5]
            hashtags = []
            for tag in tags:
                # Clean and format tag
                clean_tag = tag.replace(' ', '').replace('-', '')
                if clean_tag:
                    hashtags.append(f"#{clean_tag}")
            return ' '.join(hashtags)
        
        return ""
    
    @staticmethod
    def _trim_tweet(tweet: str, max_length: int = 280) -> str:
        """Ensure tweet fits within character limit"""
        if len(tweet) <= max_length:
            return tweet
        
        # Remove hashtags if over limit
        if "\n\n#" in tweet:
            parts = tweet.split("\n\n#")
            base_tweet = parts[0]
            if len(base_tweet) <= max_length:
                return base_tweet
        
        # Truncate description if needed
        lines = tweet.split('\n')
        while len('\n'.join(lines)) > max_length and len(lines) > 3:
            # Remove middle content lines
            lines.pop(len(lines) // 2)
        
        return '\n'.join(lines)
    
    @staticmethod
    def create_multiple_variations(post, count: int = 5) -> List[str]:
        """Create multiple tweet variations for A/B testing"""
        strategies = ["hook", "stat", "problem", "list", "question", "thread"]
        variations = []
        
        for i in range(min(count, len(strategies))):
            tweet = EnhancedTweetGenerator.create_engaging_tweet(post, strategies[i])
            variations.append({
                'strategy': strategies[i],
                'tweet': tweet,
                'length': len(tweet)
            })
        
        return variations
    
    @staticmethod
    def analyze_tweet_quality(tweet: str) -> Dict:
        """Analyze tweet for engagement factors"""
        score = 0
        feedback = []
        
        # Check for emojis
        emoji_count = sum(1 for char in tweet if ord(char) > 127000)
        if emoji_count >= 2:
            score += 10
            feedback.append("✅ Good use of emojis")
        else:
            feedback.append("⚠️ Consider adding more emojis")
        
        # Check for CTA
        if "👇" in tweet or "here" in tweet.lower():
            score += 15
            feedback.append("✅ Clear call-to-action")
        else:
            feedback.append("❌ Missing clear CTA")
        
        # Check for link
        if "https://" in tweet:
            score += 15
            feedback.append("✅ Link included")
        else:
            feedback.append("❌ No link found")
        
        # Check for hashtags
        if "#" in tweet:
            score += 10
            feedback.append("✅ Hashtags present")
        else:
            feedback.append("⚠️ No hashtags")
        
        # Check length (optimal 100-280)
        if 100 <= len(tweet) <= 280:
            score += 20
            feedback.append("✅ Optimal length")
        elif len(tweet) < 100:
            feedback.append("⚠️ Too short")
        else:
            feedback.append("❌ Too long")
        
        # Check for value proposition
        value_words = ['learn', 'discover', 'master', 'guide', 'tips', 'secrets', 'how to']
        if any(word in tweet.lower() for word in value_words):
            score += 15
            feedback.append("✅ Clear value proposition")
        else:
            feedback.append("⚠️ Unclear value")
        
        # Check for urgency/curiosity
        urgency_words = ['now', 'today', 'just', 'new', 'breaking', 'surprising']
        if any(word in tweet.lower() for word in urgency_words):
            score += 15
            feedback.append("✅ Creates urgency/curiosity")
        
        return {
            'score': score,
            'grade': 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D',
            'feedback': feedback,
            'length': len(tweet)
        }


# Example usage and testing
if __name__ == "__main__":
    # Mock post object for testing
    class MockPost:
        def __init__(self):
            self.title = "NLP Unlocked: Techniques for Human-Computer Interaction"
            self.slug = "nlp-unlocked"
            self.meta_description = "Unlock NLP secrets: discover techniques for human-computer interaction"
            self.tags = ["WebDev", "Supabase", "IoT", "VR", "LanguageModels"]
            self.twitter_hashtags = "#NLP #AI #MachineLearning"
    
    post = MockPost()
    
    print("=" * 70)
    print("ENHANCED TWEET VARIATIONS (AI-Friendly, No Personal Pronouns)")
    print("=" * 70)
    
    # Generate multiple variations
    variations = EnhancedTweetGenerator.create_multiple_variations(post, 6)
    
    for i, var in enumerate(variations, 1):
        print(f"\n📱 VARIATION {i} ({var['strategy'].upper()}) - {var['length']} chars")
        print("-" * 70)
        print(var['tweet'])
        print("-" * 70)
        
        # Analyze quality
        analysis = EnhancedTweetGenerator.analyze_tweet_quality(var['tweet'])
        print(f"\n📊 Quality Score: {analysis['score']}/100 (Grade: {analysis['grade']})")
        for item in analysis['feedback']:
            print(f"   {item}")
        print()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR YOUR BLOG")
    print("=" * 70)
    print("""
    1. 🎯 Rotate between different tweet strategies
    2. 📊 Track which strategies get most clicks
    3. ⏰ Post at optimal times (8-10am, 12-1pm, 5-6pm)
    4. 🔄 Repost top content after 2-3 weeks
    5. 💬 Engage with replies to boost visibility
    6. 📈 Use Twitter Analytics to optimize
    7. 🧵 Consider actual thread tweets for complex topics
    8. 🎨 Add images/GIFs for 2x more engagement
    
    ✅ NO "I" OR "MY" LANGUAGE - PERFECT FOR AI-GENERATED CONTENT
    """)