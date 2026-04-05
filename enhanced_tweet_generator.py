"""
EnhancedTweetGenerator v2 — Maximum impression growth toward 5M/3mo target
- Topic-specific hooks extracted from post content (no generic templates)
- Real multi-tweet thread support
- Engagement questions to drive replies (replies = algorithmic boost)
- Post-performance tracking for smart recycling
- EAT-optimised posting schedule awareness
- No personal pronouns (AI-safe)
"""

import random
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ── Timing windows (EAT = UTC+3) ────────────────────────────────────────────
# Targets: US West Coast evening + EU morning + local peak
EAT_BEST_HOURS = [7, 8, 9, 15, 16, 17, 20, 21]  # 24h EAT

# ── Engagement question bank by topic keyword ───────────────────────────────
TOPIC_QUESTIONS: Dict[str, List[str]] = {
    "ai": [
        "Which AI tool has changed your workflow the most? 👇",
        "Are you using AI in production yet? Reply with your stack 👇",
        "What's the biggest AI mistake you've seen teams make? 👇",
    ],
    "machine learning": [
        "What ML framework are you shipping with in 2026? 👇",
        "Biggest ML misconception you had to unlearn? 👇",
    ],
    "security": [
        "What's the most underrated security practice devs skip? 👇",
        "Have you ever caught a live breach? What tipped you off? 👇",
    ],
    "cloud": [
        "AWS, GCP, or Azure — which and why? 👇",
        "What cloud cost surprised you the most? 👇",
    ],
    "devops": [
        "What's your CI/CD stack right now? 👇",
        "Biggest DevOps lesson from a production incident? 👇",
    ],
    "nlp": [
        "Which NLP library do you actually ship with? 👇",
        "What NLP task took longer than expected? 👇",
    ],
    "python": [
        "One Python library that should be in every project? 👇",
        "Biggest Python gotcha that bit you in production? 👇",
    ],
    "javascript": [
        "What's your go-to JS framework in 2026? 👇",
        "TypeScript or plain JS — what are you shipping? 👇",
    ],
    "database": [
        "Postgres, MySQL, or something else? What and why? 👇",
        "Worst database migration story — go 👇",
    ],
    "default": [
        "What's your take? Drop it below 👇",
        "Have you run into this? What worked for you? 👇",
        "Agree or disagree? Reply with your experience 👇",
        "What would you add to this list? 👇",
    ],
}

# ── Hook templates with {topic} placeholder ──────────────────────────────────
# These are much more varied than the old 15-item list
HOOK_TEMPLATES: List[str] = [
    "🚨 {topic} just changed — here's what matters",
    "⚠️ The {topic} mistake costing devs weeks",
    "💡 What nobody tells you about {topic}",
    "🔥 Unpopular opinion on {topic}:",
    "📊 {topic} benchmarks that surprised us",
    "⚡ The fastest path to mastering {topic}",
    "🎯 One {topic} insight that changes everything",
    "🔐 {topic} secrets the docs don't mention",
    "🧵 Everything wrong with how people approach {topic}",
    "💰 How {topic} is saving teams thousands",
    "🤯 {topic} works differently than you think",
    "🚀 {topic} in 2026: what's actually changed",
    "⛔ Stop doing this with {topic}",
    "✅ The {topic} checklist no one shares",
    "📈 Why {topic} mastery separates seniors from juniors",
    "🔬 Deep dive: how {topic} actually works under the hood",
    "🛠️ Built something with {topic} — here's what learned",
    "⏱️ Cut {topic} setup time by 70% with this approach",
    "🧠 The mental model for {topic} that finally clicked",
    "🌍 {topic} patterns used at scale (not just tutorials)",
]

# ── Stat openers ─────────────────────────────────────────────────────────────
STAT_OPENERS: List[str] = [
    "📊 85% of teams skip this {topic} step — don't.",
    "⚡ Teams using this {topic} pattern ship 2× faster.",
    "💸 Poor {topic} decisions cost orgs $50K+ annually.",
    "📈 One {topic} change: 300% performance improvement.",
    "🔴 68% of {topic} bugs are caused by the same 3 mistakes.",
    "⏰ This {topic} approach cuts onboarding time in half.",
]

# ── CTA variants ─────────────────────────────────────────────────────────────
CTAS: List[str] = [
    "Full guide 👇",
    "Complete breakdown 👇",
    "Everything you need 👇",
    "Read the full walkthrough 👇",
    "Step-by-step here 👇",
    "Deep dive here 👇",
]


class EnhancedTweetGenerator:
    """
    Generates tweets optimised for engagement and impression growth.
    Designed for kubaik.github.io blog posts.
    """

    BASE_URL = "https://kubaik.github.io"

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def create_engaging_tweet(cls, post, strategy: str = "auto") -> str:
        strategies = ["hook", "stat", "problem", "list", "question", "thread_preview"]
        if strategy == "auto":
            # Deterministic rotation based on post slug so reruns stay consistent
            idx = int(hashlib.md5(post.slug.encode()).hexdigest(), 16) % len(strategies)
            strategy = strategies[idx]

        dispatch = {
            "hook":           cls._hook_tweet,
            "stat":           cls._stat_tweet,
            "problem":        cls._problem_tweet,
            "list":           cls._list_tweet,
            "question":       cls._question_tweet,
            "thread_preview": cls._thread_preview_tweet,
        }
        fn = dispatch.get(strategy, cls._hook_tweet)
        return fn(post)

    @classmethod
    def create_multiple_variations(cls, post, count: int = 6) -> List[Dict]:
        strategies = ["hook", "stat", "problem", "list", "question", "thread_preview"]
        return [
            {
                "strategy": s,
                "tweet": cls.create_engaging_tweet(post, s),
                "length": len(cls.create_engaging_tweet(post, s)),
            }
            for s in strategies[:count]
        ]

    @classmethod
    def create_thread(cls, post, num_tweets: int = 5) -> List[str]:
        """
        Build a real multi-tweet thread from post content.
        Returns a list of tweet strings — post them in order using the Twitter API
        with reply_to_id chaining.

        Tweet 1: Hook + promise
        Tweets 2–N-1: Key points from post (one per tweet)
        Tweet N: CTA + link
        """
        topic = cls._extract_topic(post)
        hook = cls._build_hook(topic, post.title)
        hashtags = cls._format_hashtags(post)
        url = f"{cls.BASE_URL}/{post.slug}/"

        # Extract bullet points from meta_description or title words
        points = cls._extract_key_points(post, num_tweets - 2)

        thread = []

        # Tweet 1 — hook
        t1 = f"{hook}\n\nA thread on {topic} 🧵\n\n(1/{num_tweets})"
        thread.append(cls._trim(t1))

        # Middle tweets — one point each
        for i, point in enumerate(points, start=2):
            t = f"{i}/{num_tweets}\n\n{point}"
            thread.append(cls._trim(t))

        # Final tweet — CTA
        t_last = (
            f"{num_tweets}/{num_tweets}\n\n"
            f"Full guide with examples, code snippets, and benchmarks:\n"
            f"🔗 {url}\n\n"
            f"{hashtags}"
        )
        thread.append(cls._trim(t_last))

        return thread

    @classmethod
    def analyze_tweet_quality(cls, tweet: str) -> Dict:
        score = 0
        feedback = []

        emoji_count = sum(1 for c in tweet if ord(c) > 127000)
        if emoji_count >= 2:
            score += 10
            feedback.append("✅ Good emoji use")
        else:
            feedback.append("⚠️ Add more emojis")

        if "👇" in tweet or "here" in tweet.lower():
            score += 15
            feedback.append("✅ Clear CTA")
        else:
            feedback.append("❌ Missing CTA")

        if "https://" in tweet:
            score += 15
            feedback.append("✅ Link included")
        else:
            feedback.append("❌ No link")

        if "#" in tweet:
            score += 10
            feedback.append("✅ Hashtags present")
        else:
            feedback.append("⚠️ No hashtags")

        if 100 <= len(tweet) <= 280:
            score += 20
            feedback.append("✅ Optimal length")
        elif len(tweet) < 100:
            feedback.append("⚠️ Too short")
        else:
            feedback.append("❌ Over 280 chars")

        value_words = ["learn", "discover", "master", "guide", "tip", "secret", "how"]
        if any(w in tweet.lower() for w in value_words):
            score += 15
            feedback.append("✅ Value proposition clear")
        else:
            feedback.append("⚠️ Unclear value")

        # NEW: check for engagement question
        if "?" in tweet:
            score += 15
            feedback.append("✅ Engagement question present (reply boost)")
        else:
            feedback.append("⚠️ No question — replies drive impressions")

        return {
            "score": min(score, 100),
            "grade": "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D",
            "feedback": feedback,
            "length": len(tweet),
        }

    # ── Strategy implementations ──────────────────────────────────────────────

    @classmethod
    def _hook_tweet(cls, post) -> str:
        topic = cls._extract_topic(post)
        hook = cls._build_hook(topic, post.title)
        excerpt = cls._excerpt(post, max_chars=80)
        question = cls._pick_question(post)
        cta = random.choice(CTAS)
        hashtags = cls._format_hashtags(post)
        url = f"{cls.BASE_URL}/{post.slug}/"

        parts = [hook, "", excerpt, "", f"{cta}\n🔗 {url}", "", question]
        if hashtags:
            parts += ["", hashtags]
        return cls._trim("\n".join(parts))

    @classmethod
    def _stat_tweet(cls, post) -> str:
        topic = cls._extract_topic(post)
        stat = random.choice(STAT_OPENERS).format(topic=topic)
        excerpt = cls._excerpt(post, max_chars=90)
        question = cls._pick_question(post)
        hashtags = cls._format_hashtags(post)
        url = f"{cls.BASE_URL}/{post.slug}/"

        parts = [stat, "", excerpt, "", f"Full breakdown 👇\n🔗 {url}", "", question]
        if hashtags:
            parts += ["", hashtags]
        return cls._trim("\n".join(parts))

    @classmethod
    def _problem_tweet(cls, post) -> str:
        topic = cls._extract_topic(post)
        url = f"{cls.BASE_URL}/{post.slug}/"
        question = cls._pick_question(post)
        hashtags = cls._format_hashtags(post)

        # Pull actual benefits from post rather than placeholders
        bullets = cls._extract_key_points(post, 3)
        bullet_lines = "\n".join(f"✅ {b}" for b in bullets)

        tweet = (
            f"Struggling with {topic}?\n\n"
            f"This guide covers:\n{bullet_lines}\n\n"
            f"Read here 👇\n🔗 {url}\n\n"
            f"{question}"
        )
        if hashtags:
            tweet += f"\n\n{hashtags}"
        return cls._trim(tweet)

    @classmethod
    def _list_tweet(cls, post) -> str:
        topic = cls._extract_topic(post)
        url = f"{cls.BASE_URL}/{post.slug}/"
        question = cls._pick_question(post)
        hashtags = cls._format_hashtags(post)

        points = cls._extract_key_points(post, 3)
        numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(points))

        tweet = (
            f"3 things about {topic} most guides skip:\n\n"
            f"{numbered}\n\n"
            f"Full list with examples 👇\n🔗 {url}\n\n"
            f"{question}"
        )
        if hashtags:
            tweet += f"\n\n{hashtags}"
        return cls._trim(tweet)

    @classmethod
    def _question_tweet(cls, post) -> str:
        """Lead with engagement question, follow with value."""
        topic = cls._extract_topic(post)
        url = f"{cls.BASE_URL}/{post.slug}/"
        hashtags = cls._format_hashtags(post)

        lead_questions = [
            f"What's the biggest mistake you see with {topic}?",
            f"How are you handling {topic} in 2026?",
            f"Which part of {topic} trips up your team most?",
        ]
        lead = random.choice(lead_questions)
        hook = cls._build_hook(topic, post.title)

        tweet = (
            f"{lead}\n\n"
            f"{hook}\n\n"
            f"Full guide 👇\n🔗 {url}"
        )
        if hashtags:
            tweet += f"\n\n{hashtags}"
        return cls._trim(tweet)

    @classmethod
    def _thread_preview_tweet(cls, post) -> str:
        """Tease a thread — drives people to click and watch for replies."""
        topic = cls._extract_topic(post)
        url = f"{cls.BASE_URL}/{post.slug}/"
        hashtags = cls._format_hashtags(post)
        points = cls._extract_key_points(post, 3)
        arrows = "\n".join(f"→ {p}" for p in points)
        question = cls._pick_question(post)

        tweet = (
            f"🧵 {topic}: a proper breakdown\n\n"
            f"{arrows}\n"
            f"→ Real examples + benchmarks\n\n"
            f"Full article 👇\n🔗 {url}\n\n"
            f"{question}"
        )
        if hashtags:
            tweet += f"\n\n{hashtags}"
        return cls._trim(tweet)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def _extract_topic(cls, post) -> str:
        """Pull the most meaningful topic word/phrase from tags or title."""
        if hasattr(post, "tags") and post.tags:
            # Prefer tags that are not too generic
            skip = {"the", "and", "for", "with", "from", "blog", "post", "article"}
            for tag in post.tags:
                if tag.lower() not in skip and len(tag) > 2:
                    return tag
        # Fallback: first meaningful chunk of title before colon or dash
        title = post.title
        for sep in [":", " - ", " | "]:
            if sep in title:
                return title.split(sep)[0].strip()
        # Last resort: first 3 words
        words = title.split()
        return " ".join(words[:3]) if len(words) >= 3 else title

    @classmethod
    def _build_hook(cls, topic: str, title: str) -> str:
        """Build a topic-specific hook — never the same two runs in a row."""
        template = random.choice(HOOK_TEMPLATES)
        hook = template.format(topic=topic)
        # If hook is generic, append condensed title
        if len(hook) < 40:
            hook = f"{hook}\n{title}"
        return hook

    @classmethod
    def _excerpt(cls, post, max_chars: int = 100) -> str:
        """Return a clean excerpt from meta_description."""
        desc = getattr(post, "meta_description", "") or ""
        desc = desc.strip()
        if len(desc) <= max_chars:
            return desc
        # Cut at last space before limit
        truncated = desc[:max_chars]
        last_space = truncated.rfind(" ")
        return truncated[:last_space] + "…" if last_space > 0 else truncated + "…"

    @classmethod
    def _extract_key_points(cls, post, count: int) -> List[str]:
        """
        Try to pull real key points from post content sections.
        Falls back to synthetic points from title + description.
        """
        points: List[str] = []

        # Attempt 1: find ## headings in post.content
        content = getattr(post, "content", "") or ""
        if content:
            lines = content.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("## ") and not stripped.startswith("### "):
                    heading = stripped[3:].strip()
                    # Skip intro/conclusion headings
                    skip_words = {"introduction", "conclusion", "summary", "overview", "final"}
                    if not any(w in heading.lower() for w in skip_words):
                        points.append(heading)
                if len(points) >= count:
                    break

        # Attempt 2: use tags as points
        if len(points) < count and hasattr(post, "tags") and post.tags:
            for tag in post.tags:
                if tag not in points:
                    points.append(tag)
                if len(points) >= count:
                    break

        # Attempt 3: synthetic from meta_description sentences
        if len(points) < count:
            desc = getattr(post, "meta_description", "") or ""
            for sentence in desc.split("."):
                sentence = sentence.strip()
                if sentence and sentence not in points:
                    points.append(sentence)
                if len(points) >= count:
                    break

        # Pad if still short
        topic = cls._extract_topic(post)
        generic = [
            f"Best practices for {topic}",
            f"Common pitfalls to avoid in {topic}",
            f"Real-world examples and benchmarks",
            f"Step-by-step implementation guide",
        ]
        for g in generic:
            if len(points) >= count:
                break
            if g not in points:
                points.append(g)

        return points[:count]

    @classmethod
    def _pick_question(cls, post) -> str:
        """Pick an engagement question matched to the post topic."""
        topic_lower = cls._extract_topic(post).lower()
        for keyword, questions in TOPIC_QUESTIONS.items():
            if keyword in topic_lower:
                return random.choice(questions)
        return random.choice(TOPIC_QUESTIONS["default"])

    @classmethod
    def _format_hashtags(cls, post) -> str:
        if hasattr(post, "twitter_hashtags") and post.twitter_hashtags:
            return post.twitter_hashtags
        if hasattr(post, "tags") and post.tags:
            # Max 4 hashtags — more looks spammy
            cleaned = []
            for tag in post.tags[:4]:
                clean = "".join(c for c in tag if c.isalnum())
                if clean:
                    cleaned.append(f"#{clean}")
            return " ".join(cleaned)
        return ""

    @classmethod
    def _trim(cls, tweet: str, max_length: int = 280) -> str:
        if len(tweet) <= max_length:
            return tweet
        # Drop hashtag block first
        if "\n\n#" in tweet:
            candidate = tweet.split("\n\n#")[0]
            if len(candidate) <= max_length:
                return candidate
        # Trim line by line from the middle
        lines = tweet.split("\n")
        while len("\n".join(lines)) > max_length and len(lines) > 3:
            lines.pop(len(lines) // 2)
        return "\n".join(lines)


# ── Performance tracker (call from visibility_automator) ─────────────────────

class TweetPerformanceTracker:
    """
    Tracks which tweet strategies and posts perform best.
    Persists to a JSON file and feeds the recycler.
    """

    def __init__(self, data_path: str = "tweet_performance.json"):
        self.data_path = data_path
        self._data: Dict = self._load()

    def record(self, slug: str, strategy: str, tweet_id: Optional[str] = None):
        entry = self._data.setdefault(slug, {})
        entry.setdefault("strategies", {})[strategy] = {
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "tweet_id": tweet_id,
            "impressions": 0,
            "likes": 0,
            "retweets": 0,
        }
        self._save()

    def update_metrics(self, slug: str, strategy: str,
                       impressions: int = 0, likes: int = 0, retweets: int = 0):
        try:
            entry = self._data[slug]["strategies"][strategy]
            entry["impressions"] = impressions
            entry["likes"] = likes
            entry["retweets"] = retweets
            self._save()
        except KeyError:
            pass

    def top_posts_for_recycling(self, top_n: int = 5,
                                min_days_ago: int = 21) -> List[str]:
        """Return slugs of posts worth re-tweeting (high impressions, old enough)."""
        import json, time
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_ago)
        scored: List[Tuple[int, str]] = []

        for slug, data in self._data.items():
            best_impressions = 0
            latest_post = None
            for strat, metrics in data.get("strategies", {}).items():
                posted_at_str = metrics.get("posted_at", "")
                if not posted_at_str:
                    continue
                try:
                    posted_at = datetime.fromisoformat(posted_at_str)
                    if posted_at > cutoff:
                        latest_post = posted_at
                        continue  # Too recent
                    best_impressions = max(best_impressions, metrics.get("impressions", 0))
                except ValueError:
                    pass
            if best_impressions > 0 and latest_post is None:
                scored.append((best_impressions, slug))

        scored.sort(reverse=True)
        return [slug for _, slug in scored[:top_n]]

    def _load(self) -> Dict:
        import json, os
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        import json
        with open(self.data_path, "w") as f:
            json.dump(self._data, f, indent=2)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    class MockPost:
        slug = "nlp-for-devs"
        title = "NLP for Developers: Practical Techniques in 2026"
        meta_description = (
            "Master NLP pipelines with hands-on examples. "
            "Covers tokenisation, embeddings, fine-tuning, and production deployment."
        )
        tags = ["NLP", "MachineLearning", "Python", "AI", "LLMs"]
        twitter_hashtags = ""
        content = """
## Tokenisation at scale
## Choosing the right embedding model
## Fine-tuning without GPUs
## Production deployment checklist
"""

    post = MockPost()
    print("=" * 70)
    print("TWEET VARIATIONS")
    print("=" * 70)
    for var in EnhancedTweetGenerator.create_multiple_variations(post):
        print(f"\n[{var['strategy'].upper()}] {var['length']} chars")
        print("-" * 70)
        print(var["tweet"])
        analysis = EnhancedTweetGenerator.analyze_tweet_quality(var["tweet"])
        print(f"Score: {analysis['score']}/100  Grade: {analysis['grade']}")
        print()

    print("\n" + "=" * 70)
    print("THREAD PREVIEW (5 tweets)")
    print("=" * 70)
    for i, t in enumerate(EnhancedTweetGenerator.create_thread(post, 5), 1):
        print(f"\nTweet {i}: {len(t)} chars")
        print(t)