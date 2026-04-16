"""
Enhanced Visibility Automator with Multiple Tweet Strategies
Integrates with EnhancedTweetGenerator for maximum engagement

Features:
  - post_thread()        : 2-tweet thread optimised for X algorithm
  - reply_to_trending()  : Replies to high-engagement tech tweets
  - post_at_peak_or_now(): Posts at optimal EAT timezone hours
"""

import asyncio
import datetime
import tweepy
from typing import Dict, List, Optional
from enhanced_tweet_generator import EnhancedTweetGenerator


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

PEAK_HOURS_EAT = [8, 9, 12, 17, 19, 21]

TRENDING_TECH_KEYWORDS = [
    "AI tools", "machine learning", "Python tips", "software engineering",
    "web development", "cloud computing", "cybersecurity", "DevOps",
    "TypeScript", "React", "API development", "data engineering",
    "LLM", "GPT", "open source", "startup tech", "system design",
]

MIN_AUTHOR_FOLLOWERS = 500
MAX_REPLIES_PER_RUN = 3
SEARCH_RESULT_LIMIT = 20

# Words to strip when building a hook phrase from the blog title.
_HOOK_STOP_WORDS = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is", "are",
    "with", "how", "your", "my", "our", "its", "on", "at", "by", "from",
    "this", "that", "best", "using", "guide", "complete", "introduction",
    "overview", "tutorial", "tips", "top", "ways", "actually", "really",
    "without", "beyond", "vs", "why", "when", "where", "which", "who",
    "most", "every", "what", "will", "does", "behind", "inside", "between",
    "about", "after", "before", "during", "through", "across",
    # ── NEW: strip generic filler nouns that produce bad topic phrases ──
    "secrets", "revealed", "secret", "things", "stuff", "basics",
    "fundamentals", "concepts", "ideas", "points", "steps", "facts",
    "tricks", "hacks", "methods", "techniques", "approach", "approaches",
    "solution", "solutions", "answer", "answers", "insight", "insights",
    "lesson", "lessons", "tip",
}

# ── Canonical topic overrides ────────────────────────────────────
# When the title slug contains any key (lowercase), use the mapped phrase.
# This catches titles like "Database Indexing Secrets Revealed" and ensures
# the topic phrase is always meaningful and relevant.
_TOPIC_OVERRIDES = {
    "database index":   "Database Indexing",
    "indexing":         "Database Indexing",
    "query optimiz":    "Query Optimization",
    "sql ":             "SQL Optimization",
    "redis":            "Redis",
    "kafka":            "Apache Kafka",
    "postgres":         "PostgreSQL",
    "kubernetes":       "Kubernetes",
    "docker":           "Docker",
    "system design":    "System Design",
    "machine learning": "Machine Learning",
    "deep learning":    "Deep Learning",
    "neural network":   "Neural Networks",
    "large language":   "LLMs",
    "llm":              "LLMs",
    "generative ai":    "Generative AI",
    "prompt engineer":  "Prompt Engineering",
    "rag ":             "RAG",
    "vector db":        "Vector Databases",
    "microservice":     "Microservices",
    "serverless":       "Serverless",
    "ci/cd":            "CI/CD",
    "devops":           "DevOps",
    "kubernetes":       "Kubernetes",
    "terraform":        "Terraform",
    "passive income":   "Passive Income",
    "side hustle":      "Side Hustle",
    "indie hacker":     "Indie Hacking",
    "saas":             "SaaS",
    "web performance":  "Web Performance",
    "core web vital":   "Core Web Vitals",
    "websocket":        "WebSockets",
    "graphql":          "GraphQL",
    "typescript":       "TypeScript",
    "react native":     "React Native",
    "next.js":          "Next.js",
    "nextjs":           "Next.js",
    "cybersecurity":    "Cybersecurity",
    "penetration":      "Pen Testing",
    "zero trust":       "Zero Trust",
    "rate limit":       "Rate Limiting",
    "caching":          "Caching",
    "load balanc":      "Load Balancing",
    "data pipeline":    "Data Pipelines",
    "data engineer":    "Data Engineering",
    "mlops":            "MLOps",
    "burnout":          "Developer Burnout",
    "remote work":      "Remote Work",
    "tech salary":      "Tech Salaries",
    "negotiate":        "Salary Negotiation",
}

ADDITIONAL_STOP_WORDS = {
    # Generic filler nouns that make bad topic phrases
    "secrets", "revealed", "secret", "things", "stuff", "basics",
    "fundamentals", "concepts", "ideas", "points", "steps", "facts",
    "tricks", "hacks", "methods", "techniques", "approach", "approaches",
    "solution", "solutions", "answer", "answers", "insight", "insights",
    "lesson", "lessons", "tip",
}


def _extract_topic_phrase(title: str, max_words: int = 3) -> str:
    """
    Extract a concise, meaningful topic phrase from a blog post title.

    Priority order:
      1. Canonical override from _TOPIC_OVERRIDES (most reliable)
      2. First N meaningful words after stripping _HOOK_STOP_WORDS
      3. Truncated title fallback

    Short ALL-CAPS acronyms (AI, ML, API, LLM) are always preserved.

    Examples:
      "Database Indexing Secrets Revealed"          → "Database Indexing"  (override)
      "How to Build Passive Income as a Developer"  → "Passive Income"     (override)
      "AI Tools That Write Better Code"             → "AI Tools"           (meaningful words)
      "Why Most Side Projects Fail"                 → "Side Projects"      (meaningful words)
    """
    import re

    title_lower = f" {title.lower()} "

    # 1. Check canonical overrides first
    for key, phrase in _TOPIC_OVERRIDES.items():
        if key in title_lower:
            return phrase

    # 2. Strip filler words and take first N meaningful tokens
    cleaned = re.sub(r"[^\w\s\-]", " ", title)
    words = cleaned.split()
    meaningful = []
    for w in words:
        if w.lower() in _HOOK_STOP_WORDS:
            continue
        # Always keep short ALL-CAPS acronyms: AI, ML, API, LLM …
        if w.isupper() and len(w) >= 2:
            meaningful.append(w)
        elif len(w) >= 3:
            meaningful.append(w)

    if not meaningful:
        return title[:40]

    return " ".join(meaningful[:max_words])


def _get_hashtags_for_post(post) -> str:
    """
    Reliably retrieve hashtag string from a post object.

    Checks (in order):
      1. post.twitter_hashtags  — set by blog_system.py during generation
      2. post.tags              — fallback, camelCased
      3. post.seo_keywords      — last resort

    Always returns a non-empty string for tech posts by deriving from
    topic phrase if all else fails.
    """
    # 1. Prefer the pre-built tiered hashtag string
    if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags and post.twitter_hashtags.strip():
        return post.twitter_hashtags.strip()

    # 2. Build from tags
    if hasattr(post, 'tags') and post.tags:
        clean = [
            t.replace(' ', '').replace('-', '')
            for t in post.tags
            if t and len(t.replace(' ', '').replace('-', '')) >= 2
        ]
        if clean:
            return " ".join(f"#{t}" for t in clean[:5])

    # 3. Build from seo_keywords
    if hasattr(post, 'seo_keywords') and post.seo_keywords:
        tags = []
        for kw in post.seo_keywords[:5]:
            kw = kw.strip()
            if not kw:
                continue
            words = kw.split()
            if len(words) <= 3:
                tag = "".join(w.capitalize() for w in words)
                tag = tag.replace('-', '').replace('_', '')
                if tag:
                    tags.append(f"#{tag}")
        if tags:
            return " ".join(tags[:5])

    # 4. Derive from title as absolute last resort
    if hasattr(post, 'title') and post.title:
        phrase = _extract_topic_phrase(post.title, max_words=2)
        tag = phrase.replace(' ', '')
        return f"#{tag} #Programming #SoftwareEngineering"

    return "#Programming #SoftwareEngineering #TechBlog"


# ─────────────────────────────────────────────────────────────────
# VisibilityAutomator
# ─────────────────────────────────────────────────────────────────

class VisibilityAutomator:
    def __init__(self, config):
        self.config = config
        self.twitter_client = None
        self.tweet_generator = EnhancedTweetGenerator()
        self._username = None
        self._init_twitter()

    # ─────────────────────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────────────────────

    def _init_twitter(self):
        import os
        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

        missing = [k for k, v in {
            "TWITTER_API_KEY": api_key, "TWITTER_API_SECRET": api_secret,
            "TWITTER_ACCESS_TOKEN": access_token,
            "TWITTER_ACCESS_TOKEN_SECRET": access_token_secret,
            "TWITTER_BEARER_TOKEN": bearer_token,
        }.items() if not v]

        if missing:
            print(f"⚠️  Missing Twitter credentials: {', '.join(missing)}")
            return

        try:
            self.twitter_client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key, consumer_secret=api_secret,
                access_token=access_token, access_token_secret=access_token_secret,
                wait_on_rate_limit=True,
            )
            me = self.twitter_client.get_me()
            self._username = me.data.username
            print(f"✅ Twitter API initialized as @{self._username}")
        except Exception as e:
            print(f"❌ Twitter initialization failed: {e}")
            self.twitter_client = None

    # ─────────────────────────────────────────────────────────────
    # THREAD POSTING
    # ─────────────────────────────────────────────────────────────

    def post_thread(self, post) -> Dict:
        """
        Post a 2-tweet thread:
          Tweet 1 — Hook only, no URL (X suppresses reach on link tweets)
          Tweet 2 — Reply with payoff: description + URL + tiered hashtags
        """
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized.'}

        try:
            tweets = self._build_thread_tweets(post)
            thread_ids, thread_urls, previous_id = [], [], None

            print(f"🧵 Posting thread ({len(tweets)} tweets) for: {post.title}")

            for i, tweet_text in enumerate(tweets, 1):
                kwargs = {'text': tweet_text}
                if previous_id:
                    kwargs['in_reply_to_tweet_id'] = previous_id
                response = self.twitter_client.create_tweet(**kwargs)
                tweet_id = response.data['id']
                username = self._username or "i"
                tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
                thread_ids.append(tweet_id)
                thread_urls.append(tweet_url)
                previous_id = tweet_id
                print(f"  ✅ Tweet {i}/{len(tweets)}: {tweet_url}")
                if i < len(tweets):
                    import time
                    time.sleep(2)

            return {
                'success': True, 'thread_ids': thread_ids,
                'thread_urls': thread_urls,
                'first_tweet': thread_urls[0] if thread_urls else None,
                'tweet_count': len(thread_ids),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _build_thread_tweets(self, post) -> List[str]:
        """
        2-tweet format:

        Tweet 1 — Hook only. No URL, no hashtags.
                  X suppresses reach on tweets with outbound links.
                  Hook earns replies/likes/bookmarks first.

        Tweet 2 — Reply: payoff description + blog URL (UTM-tracked) + hashtags.
                  Uses _get_hashtags_for_post() which never returns empty.
        """
        base_url = self.config.get('base_url', 'https://kubaik.github.io')
        post_url = f"{base_url}/{post.slug}"
        hook_style = self.config.get('hook_style', 'knowledge_gap')
        tracked_url = self._build_post_url(
            post_url, position=2, style=hook_style)

        # ── Hashtags — always non-empty ──────────────────────────────────────
        hashtags = _get_hashtags_for_post(post)
        print(f"  🏷️  Hashtags for thread: {hashtags}")

        # ── Topic phrase — override-aware, never "Secrets Revealed" ─────────
        topic_phrase = _extract_topic_phrase(post.title, max_words=3)
        print(f"  🎯 Topic phrase: {topic_phrase}")

        # Secondary phrases for tweet body variety
        topic_words = [
            w for w in post.title.split()
            if w.lower() not in _HOOK_STOP_WORDS and len(w) >= 2
        ]
        topic_b = " ".join(topic_words[1:3]) if len(
            topic_words) > 1 else "the fundamentals"
        topic_c = topic_words[-1] if topic_words else "performance"

        # ── Hook templates (tweet 1 — no URL, no hashtags) ──────────────────
        hook_templates = {
            'knowledge_gap': (
                f"🧵 Most people approach {topic_phrase} backwards.\n\n"
                f"They spend weeks on the wrong layer and wonder why nothing scales.\n\n"
                f"The fix is simpler than you think — but only if you understand what's actually going wrong first."
            ),
            'contrarian': (
                f"🧵 Hot take: most {topic_phrase} advice actively makes your system worse.\n\n"
                f"Not because it's wrong in theory — because it ignores what breaks in production.\n\n"
                f"Here's what actually works."
            ),
            'specific_number': (
                f"🧵 3 {topic_phrase} mistakes I see constantly — even from senior engineers:\n\n"
                f"① Skipping the boring foundation work\n"
                f"② Optimising before measuring\n"
                f"③ Ignoring failure modes until they're on fire\n\n"
                f"The third one is the silent killer."
            ),
            'pattern_interrupt': (
                f"🧵 You can tell in 5 minutes whether someone truly understands "
                f"{topic_phrase} — or just thinks they do.\n\n"
                f"The difference isn't knowledge. It's what they check first when something breaks."
            ),
        }

        hook = hook_templates.get(hook_style, hook_templates['knowledge_gap'])

        # ── Payoff reply (tweet 2 — description + URL + hashtags) ───────────
        description = post.meta_description[:150].rstrip()
        if len(post.meta_description) > 150:
            description += "…"

        payoff = (
            f"Full breakdown 👇\n\n"
            f"{description}\n\n"
            f"What's inside:\n"
            f"→ Why {topic_phrase} fails at scale\n"
            f"→ {topic_b} patterns that actually work\n"
            f"→ Real benchmarks + code\n\n"
            f"{tracked_url}\n\n"
            f"{hashtags}"
        )

        tweets = [hook, payoff]
        return [t if len(t) <= 280 else t[:277] + "..." for t in tweets]

    def _build_post_url(self, post_url: str, position: int, style: str) -> str:
        return f"{post_url}?utm_source=twitter&utm_medium=thread&utm_campaign=tweet_{position}&utm_content={style}"

    # ─────────────────────────────────────────────────────────────
    # REPLY TO TRENDING
    # ─────────────────────────────────────────────────────────────

    def reply_to_trending(self, post=None, keywords: Optional[List[str]] = None,
                          max_replies: int = MAX_REPLIES_PER_RUN) -> Dict:
        """Search recent high-engagement tech tweets and reply with value-add content."""
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized.'}

        if not keywords:
            keywords = TRENDING_TECH_KEYWORDS

        replies_posted, errors = [], []

        for keyword in keywords:
            if len(replies_posted) >= max_replies:
                break
            try:
                targets = self._find_reply_targets(keyword)
                for tweet in targets:
                    if len(replies_posted) >= max_replies:
                        break
                    if tweet.author_id and str(tweet.author_id) == str(self._get_my_id()):
                        continue
                    reply_text = self._craft_reply(tweet, keyword, post)
                    if not reply_text:
                        continue
                    response = self.twitter_client.create_tweet(
                        text=reply_text, in_reply_to_tweet_id=tweet.id)
                    reply_id = response.data['id']
                    username = self._username or "i"
                    reply_url = f"https://twitter.com/{username}/status/{reply_id}"
                    replies_posted.append({
                        'keyword': keyword, 'target_id': tweet.id, 'reply_id': reply_id,
                        'reply_url': reply_url, 'reply_preview': reply_text[:80] + "..."
                    })
                    print(f"  💬 Replied to tweet {tweet.id} → {reply_url}")
                    import time
                    time.sleep(5)
            except Exception as e:
                errors.append(f"Error for '{keyword}': {e}")
                print(f"  ⚠️  {errors[-1]}")

        return {
            'success': len(replies_posted) > 0,
            'replies_posted': replies_posted,
            'reply_count': len(replies_posted),
            'errors': errors,
        }

    def _find_reply_targets(self, keyword: str) -> List:
        try:
            response = self.twitter_client.search_recent_tweets(
                query=f'"{keyword}" lang:en -is:retweet -is:reply',
                max_results=SEARCH_RESULT_LIMIT,
                tweet_fields=['public_metrics',
                              'author_id', 'created_at', 'text'],
                expansions=['author_id'],
                user_fields=['public_metrics'],
            )
            if not response.data:
                return []
            author_followers = {}
            if response.includes and 'users' in response.includes:
                for user in response.includes['users']:
                    if hasattr(user, 'public_metrics') and user.public_metrics:
                        author_followers[user.id] = user.public_metrics.get(
                            'followers_count', 0)
            filtered = [
                t for t in response.data
                if author_followers.get(t.author_id, 0) >= MIN_AUTHOR_FOLLOWERS
            ]
            filtered.sort(
                key=lambda t: t.public_metrics.get(
                    'like_count', 0) if t.public_metrics else 0,
                reverse=True,
            )
            return filtered[:5]
        except Exception as e:
            print(f"  ⚠️  Search failed for '{keyword}': {e}")
            return []

    def _craft_reply(self, tweet, keyword: str, post=None) -> Optional[str]:
        base_url = self.config.get('base_url', 'https://kubaik.github.io')
        generic_replies = [
            f"Great point on {keyword}! One thing I'd add — consistency in the fundamentals beats chasing every new tool. Wrote a deep dive recently if helpful 👇",
            f"This is spot on. The teams that get {keyword} right share one trait: they treat it as a system, not a checklist. My full breakdown: {{url}}",
            f"Agreed. The biggest mistake I see with {keyword} is skipping the boring parts early. Costs 10× later. Patterns that actually work: {{url}}",
            f"Solid take. Would add: the 'why' behind {keyword} matters as much as the 'how'. Unpacked this with examples: {{url}}",
        ]
        idx = int(str(tweet.id)[-1]) % len(generic_replies)
        if post:
            post_url = f"{base_url}/{post.slug}"
            reply = generic_replies[idx].replace("{url}", post_url)
            hashtags = _get_hashtags_for_post(post)
            first_tag = hashtags.split()[0] if hashtags else ""
            if first_tag:
                reply += f" {first_tag}"
        else:
            reply = generic_replies[idx].replace(
                " My full breakdown: {url}", "").replace(": {{url}}", ".")
        if len(reply) > 270:
            reply = reply[:267] + "..."
        return reply

    def _get_my_id(self) -> Optional[str]:
        if not hasattr(self, '_my_id'):
            try:
                me = self.twitter_client.get_me()
                self._my_id = str(me.data.id)
            except Exception:
                self._my_id = None
        return self._my_id

    # ─────────────────────────────────────────────────────────────
    # PEAK-TIME POSTING
    # ─────────────────────────────────────────────────────────────

    def is_peak_time(self) -> bool:
        return (datetime.datetime.utcnow().hour + 3) % 24 in PEAK_HOURS_EAT

    def post_at_peak_or_now(self, post, use_thread: bool = False) -> Dict:
        if not self.is_peak_time():
            eat_hour = (datetime.datetime.utcnow().hour + 3) % 24
            print(
                f"⏰ EAT hour {eat_hour}:00 is outside peak hours {PEAK_HOURS_EAT}. Posting anyway.")
        return self.post_thread(post) if use_thread else self.post_with_best_strategy(post)

    # ─────────────────────────────────────────────────────────────
    # EXISTING METHODS
    # ─────────────────────────────────────────────────────────────

    def post_to_twitter(self, tweet_text: str = None, post=None, strategy: str = "auto") -> Dict:
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized.'}
        try:
            if tweet_text:
                final_tweet = tweet_text
            elif post:
                final_tweet = self.tweet_generator.create_engaging_tweet(
                    post, strategy)
            else:
                return {'success': False, 'error': 'Either tweet_text or post must be provided'}
            analysis = self.tweet_generator.analyze_tweet_quality(final_tweet)
            print(
                f"📊 Tweet Quality: {analysis['score']}/100 (Grade: {analysis['grade']})")
            response = self.twitter_client.create_tweet(text=final_tweet)
            tweet_id = response.data['id']
            username = self._username or "i"
            return {
                'success': True, 'tweet_id': tweet_id,
                'url': f"https://twitter.com/{username}/status/{tweet_id}",
                'tweet_text': final_tweet, 'quality_score': analysis['score'], 'strategy': strategy,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def post_with_best_strategy(self, post) -> Dict:
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized'}
        variations = self.tweet_generator.create_multiple_variations(
            post, count=5)
        best = max(variations, key=lambda v: self.tweet_generator.analyze_tweet_quality(
            v['tweet'])['score'])
        print(
            f"🎯 Strategy: {best['strategy']} | Score: {self.tweet_generator.analyze_tweet_quality(best['tweet'])['score']}")
        return self.post_to_twitter(tweet_text=best['tweet'], strategy=best['strategy'])

    def generate_tweet_preview(self, post, strategy: str = "auto") -> Dict:
        tweet = self.tweet_generator.create_engaging_tweet(post, strategy)
        analysis = self.tweet_generator.analyze_tweet_quality(tweet)
        return {
            'tweet': tweet, 'length': len(tweet), 'strategy': strategy,
            'quality_score': analysis['score'], 'grade': analysis['grade'], 'feedback': analysis['feedback'],
        }

    def generate_all_variations(self, post) -> list:
        variations = self.tweet_generator.create_multiple_variations(
            post, count=6)
        results = []
        for var in variations:
            analysis = self.tweet_generator.analyze_tweet_quality(var['tweet'])
            results.append({
                'strategy': var['strategy'], 'tweet': var['tweet'], 'length': var['length'],
                'quality_score': analysis['score'], 'grade': analysis['grade'], 'feedback': analysis['feedback'],
            })
        return sorted(results, key=lambda x: x['quality_score'], reverse=True)

    def test_twitter_connection(self) -> Dict:
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized'}
        try:
            me = self.twitter_client.get_me()
            return {'success': True, 'username': me.data.username, 'name': me.data.name, 'id': me.data.id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_social_posts(self, post) -> Dict:
        twitter_post = self.tweet_generator.create_engaging_tweet(
            post, strategy="auto")
        hashtags = _get_hashtags_for_post(post)
        linkedin_hashtags = hashtags  # already formatted as "#Tag1 #Tag2 ..."

        linkedin_post = f"""🚀 New Article: {post.title}

{post.meta_description}

In this comprehensive guide, I cover:
✅ Key concepts and fundamentals
✅ Best practices from industry leaders
✅ Real-world implementation examples
✅ Common pitfalls to avoid

Read the full article: https://kubaik.github.io/{post.slug}

{linkedin_hashtags}
"""
        return {
            'twitter':      twitter_post,
            'linkedin':     linkedin_post,
            'reddit_title': f"[Guide] {post.title}",
            'reddit':       f"{post.meta_description}\n\nFull guide: https://kubaik.github.io/{post.slug}\n\nHappy to answer questions!",
            'facebook':     f"Just published: {post.title}\n\n{post.meta_description}\n\nhttps://kubaik.github.io/{post.slug}",
        }


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    import sys

    print("=" * 70)
    print("ENHANCED VISIBILITY AUTOMATOR — TESTING")
    print("=" * 70)

    class MockPost:
        def __init__(self):
            self.title = "Database Indexing Secrets Revealed"
            self.slug = "database-indexing-secrets-revealed"
            self.meta_description = "Database indexing improves query performance by up to 90%. Learn the indexing patterns that eliminate slow queries in production PostgreSQL and MySQL."
            self.tags = ["DatabaseIndexing", "SQL",
                         "PostgreSQL", "Performance", "Backend"]
            self.seo_keywords = [
                "database indexing", "query optimization", "postgresql performance", "sql indexes"]
            # Simulate a post loaded from disk where twitter_hashtags was NOT saved
            # self.twitter_hashtags is intentionally omitted to test the fallback

    post = MockPost()

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    visibility = VisibilityAutomator(config)

    print(f"\n🔍 Topic phrase extracted: '{_extract_topic_phrase(post.title)}'")
    print(f"🏷️  Hashtags resolved:      '{_get_hashtags_for_post(post)}'")

    print("\n🧵 THREAD PREVIEW")
    print("=" * 70)
    thread_tweets = visibility._build_thread_tweets(post)
    for i, t in enumerate(thread_tweets, 1):
        print(f"\nTweet {i} ({len(t)} chars):\n{'-'*50}\n{t}")

    print("\n\n🎨 SINGLE-TWEET VARIATIONS")
    print("=" * 70)
    variations = visibility.generate_all_variations(post)
    for i, var in enumerate(variations, 1):
        print(
            f"\n📱 VARIATION {i}: {var['strategy'].upper()} | Score: {var['quality_score']}/100 | {var['length']} chars")
        print(f"{'-'*50}\n{var['tweet']}")

    if visibility.twitter_client:
        choice = input("\nPost? (1=best tweet  2=thread  3=skip): ").strip()
        if choice == "1":
            result = visibility.post_with_best_strategy(post)
            print(
                f"{'✅' if result['success'] else '❌'} {result.get('url', result.get('error'))}")
        elif choice == "2":
            result = visibility.post_thread(post)
            print(
                f"{'✅ Thread posted' if result['success'] else '❌'} ({result.get('tweet_count','')} tweets) {result.get('first_tweet', result.get('error',''))}")
    else:
        print(
            "\n⚠️  No live Twitter client. Set credentials in environment to test posting.")
