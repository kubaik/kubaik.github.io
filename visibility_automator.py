"""
Enhanced Visibility Automator with Multiple Tweet Strategies
Integrates with EnhancedTweetGenerator for maximum engagement

New features:
  - post_thread()           : Converts a blog post into a multi-tweet thread
  - reply_to_trending()     : Finds trending tech tweets and replies with value-add content
  - schedule_peak_post()    : Posts at optimal EAT timezone hours
"""

import asyncio
import datetime
import tweepy
from typing import Dict, List, Optional
from enhanced_tweet_generator import EnhancedTweetGenerator


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Peak engagement hours in East Africa Time (UTC+3)
PEAK_HOURS_EAT = [8, 9, 12, 17, 19, 21]

# Tech keywords used to find trending tweets to reply to
TRENDING_TECH_KEYWORDS = [
    "AI tools", "machine learning", "Python tips", "software engineering",
    "web development", "cloud computing", "cybersecurity", "DevOps",
    "TypeScript", "React", "API development", "data engineering",
    "LLM", "GPT", "open source", "startup tech", "system design",
]

# Minimum follower count on a tweet author before we bother replying
MIN_AUTHOR_FOLLOWERS = 500

# Max replies per run (avoid spamming)
MAX_REPLIES_PER_RUN = 3

# Max tweets to scan when looking for reply targets
SEARCH_RESULT_LIMIT = 20


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
        """Initialize Twitter API v2 client"""

        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

        missing = []
        if not api_key:
            missing.append("TWITTER_API_KEY")
        if not api_secret:
            missing.append("TWITTER_API_SECRET")
        if not access_token:
            missing.append("TWITTER_ACCESS_TOKEN")
        if not access_token_secret:
            missing.append("TWITTER_ACCESS_TOKEN_SECRET")
        if not bearer_token:
            missing.append("TWITTER_BEARER_TOKEN")

        if missing:
            print(f"⚠️  Missing Twitter credentials: {', '.join(missing)}")
            return

        try:
            self.twitter_client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
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
        Convert a blog post into a tweet thread and post it.

        Thread structure (7 tweets):
          1. Hook  — grab attention
          2. Problem  — pain point the post addresses
          3. Key insight — the core idea
          4. Tip 1  — first actionable point
          5. Tip 2  — second actionable point
          6. Tip 3  — third actionable point
          7. CTA  — link back to the full post

        Returns a dict with success status, thread_ids, and URLs.
        """
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized. Check API credentials.'}

        try:
            tweets = self._build_thread_tweets(post)
            thread_ids = []
            thread_urls = []
            previous_id = None

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

                print(f"  ✅ Tweet {i}/{len(tweets)} posted: {tweet_url}")

                # Small pause between tweets to avoid rate limits
                if i < len(tweets):
                    import time
                    time.sleep(2)

            return {
                'success':     True,
                'thread_ids':  thread_ids,
                'thread_urls': thread_urls,
                'first_tweet': thread_urls[0] if thread_urls else None,
                'tweet_count': len(thread_ids),
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _build_thread_tweets(self, post) -> List[str]:
        """
        Build a 4-tweet thread (cost optimised — was 7).
        Keeps impression value while halving API write calls.
        """
        base_url = self.config.get('base_url', 'https://kubaik.github.io')
        post_url = f"{base_url}/{post.slug}"
        short_title = post.title if len(
            post.title) <= 60 else post.title[:57] + "..."

        # Pull up to 3 shortest hashtags
        hashtags = ""
        if hasattr(post, 'tags') and post.tags:
            sorted_tags = sorted(post.tags, key=len)[:3]
            hashtags = " ".join(
                f"#{t.replace(' ', '').replace('-', '')}" for t in sorted_tags
            )

        # Derive talking points from title words
        title_words = [w for w in post.title.split() if len(w) > 4]
        topic_a = title_words[0] if len(title_words) > 0 else "this topic"
        topic_b = title_words[1] if len(title_words) > 1 else "best practices"
        topic_c = title_words[-1] if len(title_words) > 2 else "performance"

        hook_style = self.config.get('hook_style', 'knowledge_gap')

        hook_templates = {
            'knowledge_gap': (
                f"🧵 There's one thing most people skip when approaching {topic_a}.\n\n"
                f"It costs them weeks later.\n\n"
                f"{short_title} — a thread 👇\n"
                f"(full guide in the blog for those who want to go deeper)"
            ),
            'contrarian': (
                f"🧵 Most advice on {topic_a} is wrong.\n\n"
                f"Not slightly off — actively harmful.\n\n"
                f"Here's what {short_title} actually looks like when done right 👇\n"
                f"(full breakdown in the blog)"
            ),
            'specific_number': (
                f"🧵 3 things I wish I knew before spending months on {topic_a}.\n\n"
                f"{short_title} — lessons learned the hard way 👇\n"
                f"(full guide in the blog)"
            ),
            'pattern_interrupt': (
                f"🧵 You can tell within 5 minutes whether someone understands "
                f"{topic_a} or just thinks they do.\n\n"
                f"The difference is subtle. {short_title} 👇\n"
                f"(full breakdown in the blog)"
            ),
        }

        hook = hook_templates.get(hook_style, hook_templates['knowledge_gap'])

        # UTM-tagged URLs per tweet position for click attribution
        url_t2 = self._build_post_url(post_url, position=2, style=hook_style)
        url_t4 = self._build_post_url(post_url, position=4, style=hook_style)

        tweets = [
            # 1. Hook — no URL (avoids Twitter reach suppression)
            hook,

            # 2. Problem + key insight — early URL captures high-intent readers
            #    who won't wait for tweet #4
            (
                f"1/ Most people get this wrong:\n\n"
                f"{post.meta_description[:180]}\n\n"
                f"The fix starts with understanding {topic_a} properly.\n\n"
                f"Full breakdown: "
                f"{url_t4}"
                + (f"\n{hashtags}" if hashtags else "")
            ),

            # 3. Top tips
            (
                f"2/ What the best practitioners do differently:\n\n"
                f"✅ Nail {topic_a} fundamentals first\n"
                f"✅ Apply {topic_b} discipline early\n"
                f"✅ Optimise {topic_c} before it becomes expensive to fix"
            ),

            # 4. CTA — specific payoff language to increase click intent
            (
                f"3/ TL;DR — if you only read one thing on {topic_a} this week, "
                f"make it this.\n\n"
                f"Full guide with examples, code, and the mistakes to avoid 👇\n"
                f"{url_t4}"
            ),
        ]

        # Safety trim — no tweet over 280 chars
        return [t if len(t) <= 280 else t[:277] + "..." for t in tweets]

    def _build_post_url(self, post_url: str, position: int, style: str) -> str:
        """Append UTM params so each tweet's clicks are distinguishable in analytics."""
        return (
            f"{post_url}"
            f"?utm_source=twitter"
            f"&utm_medium=thread"
            f"&utm_campaign=tweet_{position}"
            f"&utm_content={style}"
        )

    # ─────────────────────────────────────────────────────────────
    # REPLY TO TRENDING
    # ─────────────────────────────────────────────────────────────

    def reply_to_trending(self, post=None, keywords: Optional[List[str]] = None,
                          max_replies: int = MAX_REPLIES_PER_RUN) -> Dict:
        """
        Search for recent high-engagement tweets on tech topics and reply
        with a value-add comment that naturally references the blog post.

        Args:
            post:        Blog post object (used to craft contextual replies).
                         Pass None to use generic replies.
            keywords:    List of search keywords. Defaults to TRENDING_TECH_KEYWORDS.
            max_replies: Maximum number of replies to post in one run.

        Returns:
            Dict with success status and list of replies posted.
        """
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized. Check API credentials.'}

        if not keywords:
            keywords = TRENDING_TECH_KEYWORDS

        replies_posted = []
        errors = []

        # Cycle through keywords until we hit max_replies
        for keyword in keywords:
            if len(replies_posted) >= max_replies:
                break

            try:
                targets = self._find_reply_targets(keyword)
                if not targets:
                    continue

                for tweet in targets:
                    if len(replies_posted) >= max_replies:
                        break

                    # Don't reply to ourselves
                    if tweet.author_id and str(tweet.author_id) == str(self._get_my_id()):
                        continue

                    reply_text = self._craft_reply(tweet, keyword, post)
                    if not reply_text:
                        continue

                    response = self.twitter_client.create_tweet(
                        text=reply_text,
                        in_reply_to_tweet_id=tweet.id
                    )

                    reply_id = response.data['id']
                    username = self._username or "i"
                    reply_url = f"https://twitter.com/{username}/status/{reply_id}"

                    replies_posted.append({
                        'keyword':       keyword,
                        'target_id':     tweet.id,
                        'reply_id':      reply_id,
                        'reply_url':     reply_url,
                        'reply_preview': reply_text[:80] + "...",
                    })

                    print(
                        f"  💬 Replied to tweet {tweet.id} with keyword '{keyword}'")
                    print(f"     Reply URL: {reply_url}")

                    # Pause between replies to look human
                    import time
                    time.sleep(5)

            except Exception as e:
                error_msg = f"Error processing keyword '{keyword}': {e}"
                errors.append(error_msg)
                print(f"  ⚠️  {error_msg}")
                continue

        return {
            'success':       len(replies_posted) > 0,
            'replies_posted': replies_posted,
            'reply_count':   len(replies_posted),
            'errors':        errors,
        }

    def _find_reply_targets(self, keyword: str) -> List:
        """
        Search for recent tweets matching the keyword.
        Filters out retweets and replies; prefers tweets with some engagement.
        """
        try:
            query = (
                f'"{keyword}" lang:en '
                f'-is:retweet -is:reply'
            )
            response = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=SEARCH_RESULT_LIMIT,
                tweet_fields=['public_metrics',
                              'author_id', 'created_at', 'text'],
                expansions=['author_id'],
                user_fields=['public_metrics'],
            )

            if not response.data:
                return []

            # Build author follower map from includes
            author_followers = {}
            if response.includes and 'users' in response.includes:
                for user in response.includes['users']:
                    if hasattr(user, 'public_metrics') and user.public_metrics:
                        author_followers[user.id] = user.public_metrics.get(
                            'followers_count', 0)

            # Filter: author must have MIN_AUTHOR_FOLLOWERS followers
            filtered = []
            for tweet in response.data:
                followers = author_followers.get(tweet.author_id, 0)
                if followers >= MIN_AUTHOR_FOLLOWERS:
                    filtered.append(tweet)

            # Sort by like count descending to target most visible tweets
            filtered.sort(
                key=lambda t: t.public_metrics.get(
                    'like_count', 0) if t.public_metrics else 0,
                reverse=True
            )
            return filtered[:5]  # Top 5 candidates per keyword

        except Exception as e:
            print(f"  ⚠️  Search failed for '{keyword}': {e}")
            return []

    def _craft_reply(self, tweet, keyword: str, post=None) -> Optional[str]:
        """
        Craft a value-add reply to a tweet.
        If a post is provided, naturally weave in the blog link.
        Replies are kept under 270 chars to be safe.
        """
        base_url = self.config.get('base_url', 'https://kubaik.github.io')

        # Generic value-add replies (rotate based on tweet id to vary tone)
        generic_replies = [
            f"Great point on {keyword}! One thing I'd add — consistency in the fundamentals beats chasing every new tool. Wrote a deep dive on this recently if helpful 👇",
            f"This is spot on. The teams that get {keyword} right share one trait: they treat it as a system, not a checklist. My full breakdown: {{url}}",
            f"Agreed. The biggest mistake I see with {keyword} is skipping the boring parts early. Costs 10× later. Wrote about the patterns that actually work: {{url}}",
            f"Solid take. Would add: the 'why' behind {keyword} matters as much as the 'how'. Unpacked this with examples here: {{url}}",
        ]

        if post:
            post_url = f"{base_url}/{post.slug}"
            idx = int(str(tweet.id)[-1]) % len(generic_replies)
            reply = generic_replies[idx].replace("{url}", post_url)

            # Add a relevant hashtag from the post
            tag = ""
            if hasattr(post, 'tags') and post.tags:
                clean = post.tags[0].replace(' ', '').replace('-', '')
                tag = f" #{clean}"

            reply = reply + tag
        else:
            idx = int(str(tweet.id)[-1]) % len(generic_replies)
            reply = generic_replies[idx].replace(
                " My full breakdown: {url}", "").replace(": {{url}}", ".")

        # Safety trim
        if len(reply) > 270:
            reply = reply[:267] + "..."

        return reply

    def _get_my_id(self) -> Optional[str]:
        """Return the authenticated user's numeric ID (cached after first call)."""
        if not hasattr(self, '_my_id'):
            try:
                me = self.twitter_client.get_me()
                self._my_id = str(me.data.id)
            except Exception:
                self._my_id = None
        return self._my_id

    # ─────────────────────────────────────────────────────────────
    # PEAK-TIME POSTING HELPER
    # ─────────────────────────────────────────────────────────────

    def is_peak_time(self) -> bool:
        """Return True if the current EAT hour is a peak engagement hour."""
        eat_hour = (datetime.datetime.utcnow().hour + 3) % 24
        return eat_hour in PEAK_HOURS_EAT

    def post_at_peak_or_now(self, post, use_thread: bool = False) -> Dict:
        """
        Post immediately if it's a peak hour, otherwise post anyway but log the warning.
        Set use_thread=True to post as a thread instead of a single tweet.
        """
        if not self.is_peak_time():
            eat_hour = (datetime.datetime.utcnow().hour + 3) % 24
            print(
                f"⏰ Current EAT hour ({eat_hour}:00) is outside peak hours {PEAK_HOURS_EAT}.")
            print(
                "   Posting anyway — consider scheduling with a cron job for peak times.")

        if use_thread:
            return self.post_thread(post)
        else:
            return self.post_with_best_strategy(post)

    # ─────────────────────────────────────────────────────────────
    # EXISTING METHODS (unchanged)
    # ─────────────────────────────────────────────────────────────

    def post_to_twitter(self, tweet_text: str = None, post=None, strategy: str = "auto") -> Dict:
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized. Check API credentials.'}

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
                f"📊 Tweet Quality Score: {analysis['score']}/100 (Grade: {analysis['grade']})")

            response = self.twitter_client.create_tweet(text=final_tweet)
            tweet_id = response.data['id']
            username = self._username or "i"

            return {
                'success':       True,
                'tweet_id':      tweet_id,
                'url':           f"https://twitter.com/{username}/status/{tweet_id}",
                'tweet_text':    final_tweet,
                'quality_score': analysis['score'],
                'strategy':      strategy,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def post_with_best_strategy(self, post) -> Dict:
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized'}

        variations = self.tweet_generator.create_multiple_variations(
            post, count=5)
        best = max(variations,
                   key=lambda v: self.tweet_generator.analyze_tweet_quality(v['tweet'])['score'])

        print(f"🎯 Selected strategy: {best['strategy']}")
        print(
            f"📊 Quality score: {self.tweet_generator.analyze_tweet_quality(best['tweet'])['score']}")

        return self.post_to_twitter(tweet_text=best['tweet'], strategy=best['strategy'])

    def generate_tweet_preview(self, post, strategy: str = "auto") -> Dict:
        tweet = self.tweet_generator.create_engaging_tweet(post, strategy)
        analysis = self.tweet_generator.analyze_tweet_quality(tweet)
        return {
            'tweet':         tweet,
            'length':        len(tweet),
            'strategy':      strategy,
            'quality_score': analysis['score'],
            'grade':         analysis['grade'],
            'feedback':      analysis['feedback'],
        }

    def generate_all_variations(self, post) -> list:
        variations = self.tweet_generator.create_multiple_variations(
            post, count=6)
        results = []
        for var in variations:
            analysis = self.tweet_generator.analyze_tweet_quality(var['tweet'])
            results.append({
                'strategy':      var['strategy'],
                'tweet':         var['tweet'],
                'length':        var['length'],
                'quality_score': analysis['score'],
                'grade':         analysis['grade'],
                'feedback':      analysis['feedback'],
            })
        return sorted(results, key=lambda x: x['quality_score'], reverse=True)

    def test_twitter_connection(self) -> Dict:
        if not self.twitter_client:
            return {'success': False, 'error': 'Twitter client not initialized'}
        try:
            me = self.twitter_client.get_me()
            return {'success': True, 'username': me.data.username,
                    'name': me.data.name, 'id': me.data.id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_social_posts(self, post) -> Dict:
        twitter_post = self.tweet_generator.create_engaging_tweet(
            post, strategy="auto")

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
        facebook_post = f"""Hey everyone! 👋

Just published a new article on {post.title}.

{post.meta_description}

If you've been curious about this topic or looking to improve your skills, this guide has you covered.

Read it here: https://kubaik.github.io/{post.slug}

Let me know what you think! 💬
"""
        return {
            'twitter':      twitter_post,
            'linkedin':     linkedin_post,
            'reddit_title': reddit_title,
            'reddit':       reddit_post,
            'facebook':     facebook_post,
        }

    def _format_hashtags_linkedin(self, post) -> str:
        if hasattr(post, 'tags') and post.tags:
            tags = post.tags[:5]
            return ' '.join(f"#{t.replace(' ', '').replace('-', '')}" for t in tags if t)
        return ""


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
            self.title = "Secure APIs: Best Practices for Protection"
            self.slug = "secure-apis"
            self.meta_description = ("Learn API security best practices to protect "
                                     "your data and prevent breaches.")
            self.tags = ["coding", "innovation",
                         "CloudNative", "OpenAPI", "5G"]
            self.twitter_hashtags = "#coding #innovation #CloudNative"

    post = MockPost()

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    visibility = VisibilityAutomator(config)

    # ── Thread preview ────────────────────────────────────────────
    print("\n🧵 THREAD PREVIEW")
    print("=" * 70)
    thread_tweets = visibility._build_thread_tweets(post)
    for i, t in enumerate(thread_tweets, 1):
        print(f"\nTweet {i} ({len(t)} chars):")
        print("-" * 50)
        print(t)

    # ── All single-tweet variations ───────────────────────────────
    print("\n\n🎨 SINGLE-TWEET VARIATIONS")
    print("=" * 70)
    variations = visibility.generate_all_variations(post)
    for i, var in enumerate(variations, 1):
        print(f"\n📱 VARIATION {i}: {var['strategy'].upper()}")
        print(f"   Score: {var['quality_score']}/100 (Grade: {var['grade']})")
        print(f"   Length: {var['length']} chars")
        print("-" * 50)
        print(var['tweet'])

    # ── Live posting (only if credentials are configured) ─────────
    if visibility.twitter_client:
        print("\n\n🔍 Twitter connection active.")

        choice = input(
            "\nWhat would you like to post?\n"
            "  1. Best single tweet\n"
            "  2. Thread\n"
            "  3. Reply to trending (dry-run search only)\n"
            "  4. Skip\n"
            "Choice: "
        ).strip()

        if choice == "1":
            result = visibility.post_with_best_strategy(post)
            if result['success']:
                print(f"✅ Tweet posted: {result['url']}")
            else:
                print(f"❌ Failed: {result['error']}")

        elif choice == "2":
            result = visibility.post_thread(post)
            if result['success']:
                print(f"✅ Thread posted ({result['tweet_count']} tweets)")
                print(f"   First tweet: {result['first_tweet']}")
            else:
                print(f"❌ Failed: {result['error']}")

        elif choice == "3":
            print("\n🔍 Searching for reply targets (no replies will be posted)...")
            for kw in TRENDING_TECH_KEYWORDS[:3]:
                targets = visibility._find_reply_targets(kw)
                print(f"\nKeyword: '{kw}' → {len(targets)} candidates")
                for t in targets[:2]:
                    preview = t.text[:80].replace('\n', ' ')
                    likes = t.public_metrics.get(
                        'like_count', 0) if t.public_metrics else 0
                    print(f"  [{likes} ❤️] {preview}...")
                    print(
                        f"  Reply would be: {visibility._craft_reply(t, kw, post)[:100]}...")
        else:
            print("Skipped.")
    else:
        print("\n⚠️  No live Twitter client. Set credentials in environment variables to test posting.")
