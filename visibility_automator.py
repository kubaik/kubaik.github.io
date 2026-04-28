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

CHANGES (2026-04-28):
  - Trending hashtag is NO LONGER fetched from the Twitter API.
  - Cache file now supports an array of hashtags under "hashtags" key.
    One tag is picked at random each run.
  - The randomly selected trending tag is NOT injected into tweet hashtags.
    It is available via _get_trending_tag() for logging/reference only.
  - fetch_daily_trending_hashtag() is kept for reference but never called.
  - Only ONE tweet is posted per run: post_single_tweet().
    post_with_best_strategy() remains as a fallback only when
    post_single_tweet() fails (called from blog_system.py).

Cache file format (.trending_hashtag_cache.json):
  {
    "date": "2026-04-28",
    "hashtags": ["#FutureOfHealth", "#AITools", "#TechTwitter", "#BuildInPublic"]
  }
  Legacy single-string format still supported:
  {"date": "2026-04-28", "hashtag": "#FutureOfHealth"}
"""

import datetime
import json
import os
import random
import re
import time
from pathlib import Path
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

# Cache file lives in the repo root. Update it manually to change the
# trending tag pool. One tag is picked at random per run for logging only —
# it is NOT injected into tweet hashtags.
_TRENDING_CACHE_FILE = Path(".trending_hashtag_cache.json")

_TECH_RELEVANCE_SIGNALS = {
    "ai", "ml", "python", "javascript", "typescript", "react", "node",
    "cloud", "aws", "gcp", "azure", "kubernetes", "docker", "devops",
    "security", "cyber", "data", "llm", "gpt", "openai", "tech", "code",
    "coding", "dev", "software", "startup", "saas", "api", "web", "mobile",
    "android", "ios", "swift", "kotlin", "rust", "golang", "java", "linux",
    "github", "programming", "engineer", "developer", "backend", "frontend",
    "database", "sql", "nosql", "redis", "kafka", "blockchain", "crypto",
}

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
    "database index":    "Database Indexing",
    "indexing":          "Database Indexing",
    "query optimiz":     "Query Optimization",
    "sql ":              "SQL Optimization",
    "redis":             "Redis",
    "kafka":             "Apache Kafka",
    "postgres":          "PostgreSQL",
    "kubernetes":        "Kubernetes",
    "docker":            "Docker",
    "system design":     "System Design",
    "machine learning":  "Machine Learning",
    "deep learning":     "Deep Learning",
    "neural network":    "Neural Networks",
    "large language":    "LLMs",
    "llm":               "LLMs",
    "generative ai":     "Generative AI",
    "prompt engineer":   "Prompt Engineering",
    "rag ":              "RAG",
    "vector db":         "Vector Databases",
    "microservice":      "Microservices",
    "serverless":        "Serverless",
    "ci/cd":             "CI/CD",
    "devops":            "DevOps",
    "terraform":         "Terraform",
    "passive income":    "Passive Income",
    "side hustle":       "Side Hustle",
    "side project":      "Side Projects",
    "indie hacker":      "Indie Hacking",
    "saas":              "SaaS",
    "web performance":   "Web Performance",
    "core web vital":    "Core Web Vitals",
    "websocket":         "WebSockets",
    "graphql":           "GraphQL",
    "typescript":        "TypeScript",
    "react native":      "React Native",
    "next.js":           "Next.js",
    "nextjs":            "Next.js",
    "cybersecurity":     "Cybersecurity",
    "penetration":       "Pen Testing",
    "zero trust":        "Zero Trust",
    "rate limit":        "Rate Limiting",
    "caching":           "Caching",
    "load balanc":       "Load Balancing",
    "data pipeline":     "Data Pipelines",
    "data engineer":     "Data Engineering",
    "mlops":             "MLOps",
    "burn out":          "Developer Burnout",
    "burnout":           "Developer Burnout",
    "remote work":       "Remote Work",
    "tech salar":        "Tech Salaries",
    "negotiate":         "Salary Negotiation",
    "ai ethics":         "AI Ethics",
    "ai tool":           "AI Tools",
    "netflix":           "Netflix Architecture",
    "concurrent stream": "Concurrent Streaming",
    "ai agent":          "AI Agents",
    "ai model":          "AI Models",
    "ai workflow":       "AI Workflows",
    "ai skill":          "AI Skills",
    "ai-powered":        "AI-Powered Apps",
    "chatgpt":           "ChatGPT",
    "openai":            "OpenAI",
    "fine-tun":          "Fine-Tuning LLMs",
    "artificial int":    "Artificial Intelligence",
}


def _extract_topic_phrase(title: str, max_words: int = 3) -> str:
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
# Trending hashtag — cache read only, for logging/reference only
# The selected tag is NOT injected into tweet hashtags.
# ─────────────────────────────────────────────────────────────────

def _load_trending_cache() -> Optional[str]:
    """
    Read the trending hashtag pool from the manually maintained cache file
    and return ONE tag chosen at random.

    IMPORTANT: The returned tag is used for logging/reference only.
    It is NOT injected into tweet hashtag blocks — tweets use only the
    hashtags derived from the post's own tags and seo_keywords.

    Supported cache formats
    ───────────────────────
    Array format (preferred):
      {
        "date": "2026-04-28",
        "hashtags": ["#FutureOfHealth", "#AITools", "#TechTwitter", "#BuildInPublic"]
      }

    Legacy single-string format (still supported):
      {"date": "2026-04-28", "hashtag": "#FutureOfHealth"}

    The "date" field is ignored. Returns None if file is missing or empty.
    """
    if not _TRENDING_CACHE_FILE.exists():
        print(
            f"ℹ️  No trending cache file found at {_TRENDING_CACHE_FILE}."
        )
        return None
    try:
        with open(_TRENDING_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Array format (preferred)
        hashtags = data.get("hashtags")
        if hashtags and isinstance(hashtags, list):
            valid = [h.strip()
                     for h in hashtags if isinstance(h, str) and h.strip()]
            if valid:
                chosen = random.choice(valid)
                if not chosen.startswith("#"):
                    chosen = f"#{chosen}"
                print(
                    f"📦 Trending cache pool ({len(valid)} tags) — "
                    f"randomly selected: {chosen} [for reference only, not used in tweet]"
                )
                return chosen
            print("ℹ️  'hashtags' array in cache is empty.")
            return None

        # Legacy single-string format
        tag = data.get("hashtag", "").strip()
        if tag:
            if not tag.startswith("#"):
                tag = f"#{tag}"
            print(f"📦 Trending tag from cache (legacy, reference only): {tag}")
            return tag

        print("ℹ️  Trending cache file contains no usable tags.")
        return None

    except Exception as e:
        print(f"⚠️  Could not read trending cache: {e}")
        return None


def _save_trending_cache(hashtags) -> None:
    """
    Persist a list of hashtags to the cache file.
    Accepts a list or a single string (wrapped into a list automatically).
    """
    try:
        if isinstance(hashtags, str):
            hashtags = [hashtags]
        with open(_TRENDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": datetime.date.today().isoformat(),
                    "hashtags": hashtags,
                },
                f,
                indent=4,
            )
        print(f"💾 Trending hashtag pool cached: {hashtags}")
    except Exception as e:
        print(f"⚠️  Could not write trending cache: {e}")


def fetch_daily_trending_hashtag(twitter_client) -> Optional[str]:
    """
    KEPT FOR REFERENCE — no longer called automatically.

    If you want to resume automatic fetching, call this manually and
    pass the result to _save_trending_cache().
    """
    if twitter_client is None:
        print("⚠️  No Twitter client — cannot fetch trending hashtag.")
        return None

    search_queries = [
        "#AI OR #MachineLearning OR #Python lang:en -is:retweet",
        "#WebDev OR #JavaScript OR #TypeScript lang:en -is:retweet",
        "#DevOps OR #CloudComputing OR #Kubernetes lang:en -is:retweet",
    ]
    hashtag_counts: Dict[str, int] = {}
    for query in search_queries:
        try:
            response = twitter_client.search_recent_tweets(
                query=query, max_results=100, tweet_fields=["text"],
            )
            if not response.data:
                continue
            for tweet in response.data:
                for match in re.findall(r"#(\w+)", tweet.text):
                    hashtag_counts[match] = hashtag_counts.get(match, 0) + 1
            time.sleep(1.5)
        except Exception as e:
            print(f"⚠️  Trending search error ({query[:40]}…): {e}")
            continue

    if not hashtag_counts:
        return None

    def _relevance_score(tag: str, freq: int) -> float:
        t = tag.lower()
        if t in _TECH_RELEVANCE_SIGNALS:
            return freq * 3
        if any(sig in t for sig in _TECH_RELEVANCE_SIGNALS):
            return freq * 2
        return 0

    scored = [
        (tag, _relevance_score(tag, freq))
        for tag, freq in hashtag_counts.items()
        if _relevance_score(tag, freq) > 0 and 3 <= len(tag) <= 30 and not tag.isdigit()
    ]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    best_tag = f"#{scored[0][0]}"
    _save_trending_cache([best_tag])
    return best_tag


# ─────────────────────────────────────────────────────────────────
# Hashtag resolver — uses post data only, never the trending cache
# ─────────────────────────────────────────────────────────────────

def _get_hashtags_for_post(post) -> str:
    """
    Build the hashtag block for a tweet using only the post's own data.

    Resolution order:
      1. post.twitter_hashtags  (set during generation, persisted to JSON)
      2. post.tags              (camelCased, deduped, max 5)
      3. post.seo_keywords      (camelCased, deduped, max 5)
      4. Title-derived fallback

    The trending cache is deliberately NOT used here — tweet hashtags
    come entirely from the post's content metadata.
    Total hashtags capped at 5 — X suppresses posts with 6+.
    """
    if (
        hasattr(post, "twitter_hashtags")
        and post.twitter_hashtags
        and post.twitter_hashtags.strip()
    ):
        # Respect the existing cap of 5
        parts = post.twitter_hashtags.strip().split()
        return " ".join(parts[:5])

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


def _build_single_tweet(
    post,
    base_url: str,
    hook_style: str = "auto",
) -> str:
    """
    Compose one high-engagement tweet for *post*.

    Hashtags come entirely from the post's own metadata.
    The trending cache tag is NOT used here.
    """
    _style_map = {
        "confession":        0,
        "contrarian":        1,
        "knowledge_gap":     1,
        "curiosity":         2,
        "pattern_interrupt": 2,
        "before_after":      3,
        "challenge":         4,
        "specific_number":   4,
        "observation":       5,
    }

    if hook_style == "auto" or hook_style not in _style_map:
        idx = hash(post.slug) % len(_TWEET_TEMPLATES)
    else:
        idx = _style_map[hook_style]

    template = _TWEET_TEMPLATES[idx]
    topic = _extract_topic_phrase(post.title, max_words=3)

    raw_desc = getattr(post, "meta_description", "") or ""
    teaser = raw_desc[:60].rstrip()
    if len(raw_desc) > 60:
        teaser = teaser.rsplit(" ", 1)[0] + "…"

    post_url = f"{base_url}/{post.slug}"
    hashtags = _get_hashtags_for_post(post)

    tweet = template.format(topic=topic, teaser=teaser,
                            url=post_url, tags=hashtags)

    if len(tweet) > 280:
        budget = 280 - len(tweet) + len(teaser)
        teaser = teaser[:max(budget - 3, 20)].rsplit(" ", 1)[0] + "…"
        tweet = template.format(
            topic=topic, teaser=teaser, url=post_url, tags=hashtags)
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."

    return tweet


# ─────────────────────────────────────────────────────────────────
# VisibilityAutomator
# ─────────────────────────────────────────────────────────────────

class VisibilityAutomator:
    def __init__(self, config):
        self.config = config
        self.tweet_generator = EnhancedTweetGenerator()
        self.twitter_client = None
        self._username = None

        # Step 1: build the Tweepy client (credentials check only, no API calls)
        self._init_twitter()

        # Step 2: read a random tag from the cache for logging purposes only.
        # This tag is NOT used in tweet content or hashtags.
        self._trending_tag: Optional[str] = _load_trending_cache()
        if self._trending_tag:
            print(
                f"ℹ️  Cache tag noted (reference only, not used in tweets): "
                f"{self._trending_tag}"
            )

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

    # ── Trending hashtag accessor (reference only) ────────────────

    def _get_trending_tag(self) -> Optional[str]:
        """Returns the tag read from cache at init time — for reference only."""
        return self._trending_tag

    # ── Single-tweet posting (primary method) ─────────────────────

    def post_single_tweet(self, post) -> Dict:
        """
        Compose and post ONE tweet for *post*.

        Hashtags are derived entirely from the post's own metadata.
        The trending cache tag is NOT included in the tweet.

        This is the only method that calls twitter_client.create_tweet().
        ENABLE_TWITTER_POSTING is checked by the caller (blog_system.py).

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
            f"📝 Tweet preview ({len(tweet_text)} chars):"
            f"\n{'-'*60}\n{tweet_text}\n{'-'*60}"
        )

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
            return {
                "success":    False,
                "error":      str(e),
                "tweet_text": tweet_text,
            }

    # ── Helpers: peak-time awareness ─────────────────────────────

    def is_peak_time(self) -> bool:
        return (datetime.datetime.utcnow().hour + 3) % 24 in PEAK_HOURS_EAT

    def post_at_peak_or_now(self, post) -> Dict:
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
                    replies_posted.append({
                        "keyword":       keyword,
                        "target_id":     tweet.id,
                        "reply_id":      reply_id,
                        "reply_url":     reply_url,
                        "reply_preview": reply_text[:80] + "...",
                    })
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
                            "followers_count", 0)
            filtered = [
                t for t in response.data
                if author_followers.get(t.author_id, 0) >= MIN_AUTHOR_FOLLOWERS
            ]
            filtered.sort(
                key=lambda t: (t.public_metrics.get(
                    "like_count", 0) if t.public_metrics else 0),
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

    # ── Existing helpers (kept for EnhancedTweetGenerator compat) ─

    def post_to_twitter(
        self, tweet_text: str = None, post=None, strategy: str = "auto"
    ) -> Dict:
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
            results.append({
                "strategy":      var["strategy"],
                "tweet":         var["tweet"],
                "length":        var["length"],
                "quality_score": analysis["score"],
                "grade":         analysis["grade"],
                "feedback":      analysis["feedback"],
            })
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
    print(f"🏷️  Hashtags      : '{_get_hashtags_for_post(post)}'")
    print(
        f"ℹ️  Cache tag     : '{visibility._get_trending_tag()}' (reference only, not in tweet)")

    print("\n📱 ALL 6 HOOK TEMPLATES")
    print("=" * 70)
    for i, _ in enumerate(_TWEET_TEMPLATES):
        topic = _extract_topic_phrase(post.title)
        teaser = post.meta_description[:60]
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
