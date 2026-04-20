"""
Visibility Automator — Single-Tweet Edition
Posts one high-engagement tweet per blog post, optimised for impressions
and click-throughs from the X / Twitter algorithm.

Hook philosophy
───────────────
  • Sound like a person, not a scheduler.
  • Open with tension, a number, or a confession — not a product pitch.
  • The link + hashtags sit at the END, never in the hook line.
  • Every template is tested to stay under 280 chars with a typical title.

FIXES (2026-04-16):
  - Removed post_thread() / _build_thread_tweets() entirely.
  - Added post_single_tweet() with 6 rotating human-feeling hook templates.
  - _get_hashtags_for_post() deduplicates, caps at 5, never double-hashes.
  - _TOPIC_OVERRIDES covers AI Ethics, AI Tools, Side Projects, etc.
  - Year tokens (2024-2026) stripped from topic phrases.
"""

import datetime
import os
import re
import time
from typing import Dict, List, Optional

import tweepy

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

# ─────────────────────────────────────────────────────────────────
# Stop-word set for topic-phrase extraction
# ─────────────────────────────────────────────────────────────────

_HOOK_STOP_WORDS = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is", "are",
    "with", "how", "your", "my", "our", "its", "on", "at", "by", "from",
    "this", "that", "best", "using", "guide", "complete", "introduction",
    "overview", "tutorial", "tips", "top", "ways", "actually", "really",
    "without", "beyond", "vs", "why", "when", "where", "which", "who",
    "most", "every", "what", "will", "does", "behind", "inside", "between",
    "about", "after", "before", "during", "through", "across",
    "secrets", "revealed", "secret", "things", "stuff", "basics",
    "fundamentals", "concepts", "ideas", "points", "steps", "facts",
    "tricks", "hacks", "methods", "techniques", "approach", "approaches",
    "solution", "solutions", "answer", "answers", "insight", "insights",
    "lesson", "lessons", "tip",
    "big", "new", "old", "bad", "good", "great", "real", "true", "key",
    "main", "full", "last", "next", "part", "each", "both", "many", "much",
    "more", "less", "few", "own", "same", "other", "another", "such",
    "sure", "just", "also", "even", "still", "yet", "well", "back",
    "dark", "side", "deep", "fast", "slow", "hard", "easy", "smart",
    "hidden", "ultimate", "simple", "practical", "essential", "advanced",
    "modern", "wrong", "right", "never", "always", "common",
    "say", "says", "fail", "fails", "work", "works", "make", "makes",
    "get", "gets", "know", "use", "need", "want", "find", "give", "take",
    "show", "tell", "look", "come", "keep", "let", "put", "think", "help",
    "earn", "wins", "win", "lose", "beat", "buy", "sell", "run", "start",
    "people", "person", "developer", "developers", "engineer", "engineers",
    "company", "companies", "team", "teams", "user", "users", "way",
}

# ─────────────────────────────────────────────────────────────────
# Canonical topic-phrase overrides
# ─────────────────────────────────────────────────────────────────

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
    "terraform":        "Terraform",
    "passive income":   "Passive Income",
    "side hustle":      "Side Hustle",
    "side project":     "Side Projects",
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
    "burn out":          "Developer Burnout",
    "burnout":           "Developer Burnout",
    "remote work":       "Remote Work",
    "tech salar":       "Tech Salaries",
    "negotiate":        "Salary Negotiation",
    "ai ethics":        "AI Ethics",
    "ai tool":          "AI Tools",
    "netflix":          "Netflix Architecture",
    "concurrent stream": "Concurrent Streaming",
    "ai agent":         "AI Agents",
    "ai model":         "AI Models",
    "ai workflow":      "AI Workflows",
    "ai skill":         "AI Skills",
    "ai-powered":       "AI-Powered Apps",
    "chatgpt":          "ChatGPT",
    "openai":           "OpenAI",
    "fine-tun":         "Fine-Tuning LLMs",
    "artificial int":   "Artificial Intelligence",
}


def _extract_topic_phrase(title: str, max_words: int = 3) -> str:
    """
    Return a concise, meaningful topic phrase from a blog post title.

    Priority:
      1. Canonical override  (most reliable)
      2. First N meaningful words after stop-word + year filtering
      3. Truncated title fallback
    """
    title_lower = f" {title.lower()} "
    for key, phrase in _TOPIC_OVERRIDES.items():
        if key in title_lower:
            return phrase

    cleaned = re.sub(r"[^\w\s\-]", " ", title)
    words = cleaned.split()
    meaningful: List[str] = []
    for w in words:
        if w.lower() in _HOOK_STOP_WORDS:
            continue
        if re.match(r"^\d{4}$", w):
            continue
        if w.isupper() and len(w) >= 2:
            meaningful.append(w)
        elif len(w) >= 3:
            meaningful.append(w)

    if not meaningful:
        return title[:40]
    return " ".join(meaningful[:max_words])


# ─────────────────────────────────────────────────────────────────
# Hashtag resolver
# ─────────────────────────────────────────────────────────────────

def _get_hashtags_for_post(post) -> str:
    """
    Reliably retrieve a hashtag string from a post object.

    Resolution order:
      1. post.twitter_hashtags  (set during generation, persisted to JSON)
      2. post.tags              (camelCased, deduped, max 5)
      3. post.seo_keywords      (camelCased, deduped, max 5)
      4. Title-derived fallback

    Never produces ##DoublePound or more than 5 tags (X suppresses 6+).
    """
    if (
        hasattr(post, "twitter_hashtags")
        and post.twitter_hashtags
        and post.twitter_hashtags.strip()
    ):
        return post.twitter_hashtags.strip()

    if hasattr(post, "tags") and post.tags:
        seen: set = set()
        clean: List[str] = []
        for t in post.tags:
            if not t:
                continue
            raw = t.lstrip("#").replace(" ", "").replace("-", "")
            if len(raw) < 2:
                continue
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(f"#{raw}")
            if len(clean) == 5:
                break
        if clean:
            return " ".join(clean)

    if hasattr(post, "seo_keywords") and post.seo_keywords:
        seen = set()
        tags: List[str] = []
        for kw in post.seo_keywords[:8]:
            kw = kw.strip()
            if not kw:
                continue
            parts = kw.split()
            if len(parts) <= 3:
                tag = "".join(w.capitalize() for w in parts)
                tag = re.sub(r"[^\w]", "", tag)
                if tag and tag.lower() not in seen:
                    seen.add(tag.lower())
                    tags.append(f"#{tag}")
            if len(tags) == 5:
                break
        if tags:
            return " ".join(tags)

    if hasattr(post, "title") and post.title:
        phrase = _extract_topic_phrase(post.title, max_words=2)
        tag = phrase.replace(" ", "")
        return f"#{tag} #Programming #SoftwareEngineering"

    return "#Programming #SoftwareEngineering #TechBlog"


# ─────────────────────────────────────────────────────────────────
# Single-tweet hook templates
# ─────────────────────────────────────────────────────────────────
#
# Rules for every template:
#   • First line = the hook — must create tension, curiosity, or mild
#     controversy without sounding like ad copy.
#   • Middle = one concrete promise (what they'll learn / gain).
#   • End = link + hashtags — ALWAYS last, never in the hook.
#   • Total target: 240–270 chars so there's buffer for long slugs.
#   • Placeholders:  {topic}  {teaser}  {url}  {tags}
#
# Six templates rotate by (post_id % 6) so consecutive posts feel varied.
# ─────────────────────────────────────────────────────────────────

# Design constraint: each template must stay ≤ 280 chars when rendered with:
#   topic  ≤ 25 chars  (enforced by _extract_topic_phrase max_words=3)
#   teaser ≤ 60 chars  (enforced inside _build_single_tweet)
#   url    ≤ 60 chars  (e.g. https://kubaik.github.io/some-slug-here-50ch)
#   tags   ≤ 50 chars  (5 × ~10 chars each)
# Fixed text per template is ≤ 85 chars, giving comfortable headroom.
_TWEET_TEMPLATES = [
    # 0 — confession
    "Spent months doing {topic} wrong.\n\n{teaser}\n\nWrote it down so you skip my mistakes 👇\n{url}\n\n{tags}",
    # 1 — uncomfortable truth
    "Hot take: most {topic} guides skip the part that actually matters.\n\n{teaser}\n\nReal breakdown + numbers 👇\n{url}\n\n{tags}",
    # 2 — dev vs prod curiosity gap
    "{topic} works in dev. Breaks silently in prod. Why?\n\n{teaser}\n\nI dug in. Here's what I found 👇\n{url}\n\n{tags}",
    # 3 — before / after
    "Before: days debugging {topic}.\nAfter: checklist that catches 90% of issues.\n\n{teaser}\n\nChecklist + guide 👇\n{url}\n\n{tags}",
    # 4 — direct challenge
    "Can't explain {topic} simply? You don't fully get it yet.\n\n{teaser}\n\nPlain-English guide + code 👇\n{url}\n\n{tags}",
    # 5 — observation
    "Senior devs who nail {topic} all do one thing differently.\n\n{teaser}\n\nTook me too long to notice 👇\n{url}\n\n{tags}",
]


def _build_single_tweet(post, base_url: str, hook_style: str = "auto") -> str:
    """
    Compose one high-engagement tweet for *post*.

    hook_style:
      "auto"       → rotate by slug hash (consistent per post, varied across posts)
      "confession" → template 0
      "contrarian" → template 1
      "curiosity"  → template 2
      "before_after"→ template 3
      "challenge"  → template 4
      "observation"→ template 5
      "knowledge_gap", "specific_number", "pattern_interrupt"
                   → mapped to closest template for back-compat
    """
    _style_map = {
        "confession":     0,
        "contrarian":     1,
        "knowledge_gap":  1,
        "curiosity":      2,
        "pattern_interrupt": 2,
        "before_after":   3,
        "challenge":      4,
        "specific_number": 4,
        "observation":    5,
    }

    if hook_style == "auto" or hook_style not in _style_map:
        # Deterministic rotation: same post always gets same template
        idx = hash(post.slug) % len(_TWEET_TEMPLATES)
    else:
        idx = _style_map[hook_style]

    template = _TWEET_TEMPLATES[idx]

    topic = _extract_topic_phrase(post.title, max_words=3)

    # Build a short teaser from meta_description (≤ 60 chars, ends cleanly)
    # 60 chars gives all 6 templates comfortable headroom under 280.
    raw_desc = getattr(post, "meta_description", "") or ""
    teaser = raw_desc[:60].rstrip()
    if len(raw_desc) > 60:
        teaser = teaser.rsplit(" ", 1)[0] + "…"

    post_url = f"{base_url}/{post.slug}"
    hashtags = _get_hashtags_for_post(post)

    tweet = template.format(
        topic=topic,
        teaser=teaser,
        url=post_url,
        tags=hashtags,
    )

    # Hard cap — X rejects anything over 280 chars
    if len(tweet) > 280:
        # Shorten teaser first
        budget = 280 - len(tweet) + len(teaser)
        teaser = teaser[:max(budget - 3, 20)].rsplit(" ", 1)[0] + "…"
        tweet = template.format(
            topic=topic, teaser=teaser, url=post_url, tags=hashtags
        )
        # Last resort: hard truncate
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."

    return tweet


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

    # ── Init ──────────────────────────────────────────────────────

    def _init_twitter(self):
        api_key = os.getenv("TWITTER_API_KEY")
        api_secret = os.getenv("TWITTER_API_SECRET")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

        missing = [
            k for k, v in {
                "TWITTER_API_KEY":             api_key,
                "TWITTER_API_SECRET":          api_secret,
                "TWITTER_ACCESS_TOKEN":        access_token,
                "TWITTER_ACCESS_TOKEN_SECRET": access_token_secret,
                "TWITTER_BEARER_TOKEN":        bearer_token,
            }.items()
            if not v
        ]
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
                wait_on_rate_limit=True,
            )
            me = self.twitter_client.get_me()
            self._username = me.data.username
            print(f"✅ Twitter API initialized as @{self._username}")
        except Exception as e:
            print(f"❌ Twitter initialization failed: {e}")
            self.twitter_client = None

    # ── Single-tweet posting (primary method) ─────────────────────

    def post_single_tweet(self, post) -> Dict:
        """
        Compose and post ONE engaging tweet for *post*.

        Returns:
            {
                'success': bool,
                'tweet_id': str,
                'url': str,
                'tweet_text': str,
                'char_count': int,
            }
        """
        if not self.twitter_client:
            return {"success": False, "error": "Twitter client not initialized."}

        base_url = self.config.get("base_url", "https://kubaik.github.io")
        hook_style = self.config.get("hook_style", "auto")

        tweet_text = _build_single_tweet(post, base_url, hook_style)

        print(
            f"📝 Tweet preview ({len(tweet_text)} chars):\n{'-'*60}\n{tweet_text}\n{'-'*60}")

        try:
            response = self.twitter_client.create_tweet(text=tweet_text)
            tweet_id = response.data["id"]
            username = self._username or "i"
            url = f"https://twitter.com/{username}/status/{tweet_id}"
            print(f"✅ Tweet posted: {url}")
            return {
                "success":    True,
                "tweet_id":   tweet_id,
                "url":        url,
                "tweet_text": tweet_text,
                "char_count": len(tweet_text),
            }
        except Exception as e:
            print(f"❌ Tweet failed: {e}")
            return {"success": False, "error": str(e), "tweet_text": tweet_text}

    # ── Helpers: peak-time awareness ─────────────────────────────

    def is_peak_time(self) -> bool:
        """True if current EAT hour is in PEAK_HOURS_EAT."""
        return (datetime.datetime.utcnow().hour + 3) % 24 in PEAK_HOURS_EAT

    def post_at_peak_or_now(self, post) -> Dict:
        """Post immediately; print a note if outside peak hours."""
        if not self.is_peak_time():
            eat_hour = (datetime.datetime.utcnow().hour + 3) % 24
            print(
                f"⏰ EAT hour {eat_hour}:00 is outside peak {PEAK_HOURS_EAT}. Posting anyway.")
        return self.post_single_tweet(post)

    # ── Reply to trending ─────────────────────────────────────────

    def reply_to_trending(
        self,
        post=None,
        keywords: Optional[List[str]] = None,
        max_replies: int = MAX_REPLIES_PER_RUN,
    ) -> Dict:
        if not self.twitter_client:
            return {"success": False, "error": "Twitter client not initialized."}
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
                        text=reply_text, in_reply_to_tweet_id=tweet.id
                    )
                    reply_id = response.data["id"]
                    username = self._username or "i"
                    reply_url = f"https://twitter.com/{username}/status/{reply_id}"
                    replies_posted.append(
                        {
                            "keyword":       keyword,
                            "target_id":     tweet.id,
                            "reply_id":      reply_id,
                            "reply_url":     reply_url,
                            "reply_preview": reply_text[:80] + "...",
                        }
                    )
                    print(f"  💬 Replied to tweet {tweet.id} → {reply_url}")
                    time.sleep(5)
            except Exception as e:
                errors.append(f"Error for '{keyword}': {e}")
                print(f"  ⚠️  {errors[-1]}")

        return {
            "success":        len(replies_posted) > 0,
            "replies_posted": replies_posted,
            "reply_count":    len(replies_posted),
            "errors":         errors,
        }

    def _find_reply_targets(self, keyword: str) -> List:
        try:
            response = self.twitter_client.search_recent_tweets(
                query=f'"{keyword}" lang:en -is:retweet -is:reply',
                max_results=SEARCH_RESULT_LIMIT,
                tweet_fields=["public_metrics",
                              "author_id", "created_at", "text"],
                expansions=["author_id"],
                user_fields=["public_metrics"],
            )
            if not response.data:
                return []

            author_followers: Dict = {}
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    if hasattr(user, "public_metrics") and user.public_metrics:
                        author_followers[user.id] = user.public_metrics.get(
                            "followers_count", 0
                        )

            filtered = [
                t
                for t in response.data
                if author_followers.get(t.author_id, 0) >= MIN_AUTHOR_FOLLOWERS
            ]
            filtered.sort(
                key=lambda t: (
                    t.public_metrics.get(
                        "like_count", 0) if t.public_metrics else 0
                ),
                reverse=True,
            )
            return filtered[:5]
        except Exception as e:
            print(f"  ⚠️  Search failed for '{keyword}': {e}")
            return []

    def _craft_reply(self, tweet, keyword: str, post=None) -> Optional[str]:
        base_url = self.config.get("base_url", "https://kubaik.github.io")
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
            reply = (
                generic_replies[idx]
                .replace(" My full breakdown: {url}", "")
                .replace(": {{url}}", ".")
            )
        return reply[:267] + "..." if len(reply) > 270 else reply

    def _get_my_id(self) -> Optional[str]:
        if not hasattr(self, "_my_id"):
            try:
                me = self.twitter_client.get_me()
                self._my_id = str(me.data.id)
            except Exception:
                self._my_id = None
        return self._my_id

    # ── Existing single-post helpers (kept for EnhancedTweetGenerator compat) ──

    def post_to_twitter(
        self, tweet_text: str = None, post=None, strategy: str = "auto"
    ) -> Dict:
        """Low-level: post a raw string, or let EnhancedTweetGenerator build one."""
        if not self.twitter_client:
            return {"success": False, "error": "Twitter client not initialized."}
        try:
            if tweet_text:
                final_tweet = tweet_text
            elif post:
                final_tweet = self.tweet_generator.create_engaging_tweet(
                    post, strategy)
            else:
                return {"success": False, "error": "tweet_text or post required"}

            analysis = self.tweet_generator.analyze_tweet_quality(final_tweet)
            print(
                f"📊 Tweet Quality: {analysis['score']}/100 (Grade: {analysis['grade']})")

            response = self.twitter_client.create_tweet(text=final_tweet)
            tweet_id = response.data["id"]
            username = self._username or "i"
            return {
                "success":       True,
                "tweet_id":      tweet_id,
                "url":           f"https://twitter.com/{username}/status/{tweet_id}",
                "tweet_text":    final_tweet,
                "quality_score": analysis["score"],
                "strategy":      strategy,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def post_with_best_strategy(self, post) -> Dict:
        """Pick the highest-scoring variation from EnhancedTweetGenerator and post it."""
        if not self.twitter_client:
            return {"success": False, "error": "Twitter client not initialized"}
        variations = self.tweet_generator.create_multiple_variations(
            post, count=5)
        best = max(
            variations,
            key=lambda v: self.tweet_generator.analyze_tweet_quality(v["tweet"])[
                "score"],
        )
        score = self.tweet_generator.analyze_tweet_quality(best["tweet"])[
            "score"]
        print(f"🎯 Strategy: {best['strategy']} | Score: {score}")
        return self.post_to_twitter(tweet_text=best["tweet"], strategy=best["strategy"])

    def generate_tweet_preview(self, post, strategy: str = "auto") -> Dict:
        tweet = self.tweet_generator.create_engaging_tweet(post, strategy)
        analysis = self.tweet_generator.analyze_tweet_quality(tweet)
        return {
            "tweet":         tweet,
            "length":        len(tweet),
            "strategy":      strategy,
            "quality_score": analysis["score"],
            "grade":         analysis["grade"],
            "feedback":      analysis["feedback"],
        }

    def generate_all_variations(self, post) -> list:
        variations = self.tweet_generator.create_multiple_variations(
            post, count=6)
        results = []
        for var in variations:
            analysis = self.tweet_generator.analyze_tweet_quality(var["tweet"])
            results.append(
                {
                    "strategy":      var["strategy"],
                    "tweet":         var["tweet"],
                    "length":        var["length"],
                    "quality_score": analysis["score"],
                    "grade":         analysis["grade"],
                    "feedback":      analysis["feedback"],
                }
            )
        return sorted(results, key=lambda x: x["quality_score"], reverse=True)

    def test_twitter_connection(self) -> Dict:
        if not self.twitter_client:
            return {"success": False, "error": "Twitter client not initialized"}
        try:
            me = self.twitter_client.get_me()
            return {
                "success":  True,
                "username": me.data.username,
                "name":     me.data.name,
                "id":       me.data.id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_social_posts(self, post) -> Dict:
        twitter_post = self.tweet_generator.create_engaging_tweet(
            post, strategy="auto")
        hashtags = _get_hashtags_for_post(post)
        linkedin_post = (
            f"🚀 New Article: {post.title}\n\n"
            f"{post.meta_description}\n\n"
            f"In this guide:\n"
            f"✅ Key concepts and fundamentals\n"
            f"✅ Best practices from industry experience\n"
            f"✅ Real-world implementation examples\n"
            f"✅ Common pitfalls to avoid\n\n"
            f"Read the full article: https://kubaik.github.io/{post.slug}\n\n"
            f"{hashtags}\n"
        )
        return {
            "twitter":      twitter_post,
            "linkedin":     linkedin_post,
            "reddit_title": f"[Guide] {post.title}",
            "reddit": (
                f"{post.meta_description}\n\n"
                f"Full guide: https://kubaik.github.io/{post.slug}\n\n"
                f"Happy to answer questions!"
            ),
            "facebook": (
                f"Just published: {post.title}\n\n"
                f"{post.meta_description}\n\n"
                f"https://kubaik.github.io/{post.slug}"
            ),
        }


# ─────────────────────────────────────────────────────────────────
# CLI entry point — quick smoke-test / preview
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    print("=" * 70)
    print("VISIBILITY AUTOMATOR — SINGLE TWEET PREVIEW")
    print("=" * 70)

    class MockPost:
        title = "AI Ethics: The Hidden Dangers of Relying on AI"
        slug = "ai-ethics-hidden-dangers"
        meta_description = (
            "AI ethics issues in Big Tech are rarely discussed publicly. "
            "This guide covers algorithmic harms and the internal debates "
            "that never reach the public."
        )
        tags = ["AI", "AIEthics", "Tech",
                "MachineLearning", "SoftwareEngineering"]
        seo_keywords = ["ai ethics", "big tech problems",
                        "responsible ai", "ai bias"]
        twitter_hashtags = "#AI #AIEthics #MachineLearning #Tech #GenerativeAI"

    post = MockPost()

    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    visibility = VisibilityAutomator(config)

    base_url = config.get("base_url", "https://kubaik.github.io")
    print(f"\n🔍 Topic phrase : '{_extract_topic_phrase(post.title)}'")
    print(f"🏷️  Hashtags     : '{_get_hashtags_for_post(post)}'")

    print("\n📱 ALL 6 HOOK TEMPLATES")
    print("=" * 70)
    for i, tmpl in enumerate(_TWEET_TEMPLATES):
        tweet = _build_single_tweet.__wrapped__(post, base_url) if hasattr(
            _build_single_tweet, "__wrapped__") else None
        # Direct render for preview
        topic = _extract_topic_phrase(post.title)
        teaser = post.meta_description[:100]
        url = f"{base_url}/{post.slug}"
        tags = _get_hashtags_for_post(post)
        rendered = _TWEET_TEMPLATES[i].format(
            topic=topic, teaser=teaser, url=url, tags=tags)
        print(f"\n--- Template {i} ({len(rendered)} chars) ---\n{rendered}")

    print("\n\n🎯 SELECTED TWEET (auto-rotation by slug hash)")
    print("=" * 70)
    selected = _build_single_tweet(post, base_url, hook_style="auto")
    print(f"\n{selected}\n\n({len(selected)} chars)")

    if visibility.twitter_client:
        choice = input("\nPost this tweet? (y/n): ").strip().lower()
        if choice == "y":
            result = visibility.post_single_tweet(post)
            if result["success"]:
                print(f"✅ Posted: {result['url']}")
            else:
                print(f"❌ Failed: {result['error']}")
    else:
        print("\n⚠️  No live Twitter client — set env vars to test live posting.")
