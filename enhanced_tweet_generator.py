"""
Enhanced Twitter/X Post Generator - Impression Maximization Edition
Strategies: Threads, Polls, Viral Hooks, Optimal Timing, Hot Takes
Target: 5M+ impressions for X Creator Monetization

Key improvements over original:
- Thread generation (3-5x more impressions than single tweets)
- Poll tweets (2-3x more engagement via forced interaction)
- Hot take / controversy hooks (highest engagement category)
- Stat-shock format (triggers shares from people who disagree)
- Optimal posting time guidance per EAT timezone
- Impression multiplier scoring per strategy
"""

import random
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────
# HOOK LIBRARIES
# ─────────────────────────────────────────────────────────────────

TIER1_HOOKS = [
    "🔥 Unpopular opinion:",
    "⚠️ Nobody talks about this:",
    "💀 This is killing developer productivity:",
    "🚨 Stop doing this immediately:",
    "❌ {topic} advice that's dead wrong:",
    "🤯 This breaks everything you know about {topic}:",
    "🧵 Studied {topic} for 3 months. Here's what no one tells you:",
    "🔍 The dirty secret about {topic}:",
    "📉 Why 90% of developers fail at {topic}:",
    "🎯 The one {topic} mistake costing companies millions:",
    "📊 After analyzing 500+ {topic} implementations:",
    "🏆 Top engineers don't talk about {topic} like this:",
    "⏰ {topic} is changing in 2026. Are you ready?",
    "🚀 The {topic} skill that's suddenly worth $200k+:",
    "📈 {topic} just got 10x more important. Here's why:",
]

TIER2_HOOKS = [
    "🧠 Brain dump on {topic}:",
    "💎 Hidden gem in {topic} most devs ignore:",
    "⚡ {topic} in 60 seconds (bookmark this):",
    "🔓 Unlock {topic} mastery with this framework:",
    "📚 Everything about {topic} in one post:",
    "🎓 {topic} explained like you're 5:",
    "🛠️ The {topic} toolkit that changes everything:",
    "📌 Pinning this {topic} guide (you'll need it):",
]

CTAS = [
    "Full breakdown 👇 (bookmark this)",
    "Complete guide 👇 (save for later)",
    "Deep dive here 👇",
    "Everything you need 👇",
    "Step-by-step walkthrough 👇",
]

POLL_TEMPLATES = [
    {
        "question": "What's your biggest {topic} challenge?",
        "options": ["Just getting started", "Scaling issues", "Performance", "Security"]
    },
    {
        "question": "How do you handle {topic} in production?",
        "options": ["Custom solution", "Third-party tool", "Cloud managed", "Still figuring out"]
    },
    {
        "question": "Best approach for {topic}?",
        "options": ["Build from scratch", "Use frameworks", "Managed services", "Depends on project"]
    },
    {
        "question": "How long to master {topic}?",
        "options": ["< 1 month", "1–3 months", "3–6 months", "Still learning!"]
    },
]


class EnhancedTweetGenerator:
    """Generate high-impression tweets, threads, and polls for X monetization"""

    BASE_URL = "https://kubaik.github.io"

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def create_engaging_tweet(post, strategy: str = "auto") -> str:
        strategies = ["viral_hook", "thread_preview", "poll_bait",
                      "stat_shock", "list_tease", "question", "hot_take"]
        if strategy == "auto":
            strategy = random.choices(
                strategies,
                weights=[25, 20, 15, 15, 10, 10, 5],
                k=1
            )[0]

        dispatch = {
            "viral_hook":     EnhancedTweetGenerator._viral_hook_tweet,
            "thread_preview": EnhancedTweetGenerator._thread_preview_tweet,
            "poll_bait":      EnhancedTweetGenerator._poll_bait_tweet,
            "stat_shock":     EnhancedTweetGenerator._stat_shock_tweet,
            "list_tease":     EnhancedTweetGenerator._list_tease_tweet,
            "question":       EnhancedTweetGenerator._question_tweet,
            "hot_take":       EnhancedTweetGenerator._hot_take_tweet,
        }
        return dispatch.get(strategy, EnhancedTweetGenerator._viral_hook_tweet)(post)

    @staticmethod
    def create_thread(post) -> List[str]:
        """
        Generate a full 8-tweet thread.
        Threads get 3-5x more impressions than single tweets.
        Returns list of tweet strings to post in sequence.
        """
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        hook = random.choice(TIER1_HOOKS).replace("{topic}", topic)

        tweets = [
            f"{hook}\n\n{post.title}\n\nA thread 🧵\n\n{hashtags}",

            f"2/ The problem:\n\nMost developers approach {topic} the wrong way.\n\n"
            f"They jump straight to implementation without understanding the fundamentals.\n\n"
            f"Here's what they miss 👇",

            f"3/ Key insight:\n\n{post.meta_description}\n\n"
            f"This single shift changes everything about how you approach {topic}.",

            f"4/ The framework that works:\n\n"
            f"✅ Start with the right mental model\n"
            f"✅ Validate your approach early\n"
            f"✅ Iterate based on real feedback\n"
            f"✅ Optimize only after it works\n\n"
            f"(Most devs skip to step 4. That's why they fail.)",

            f"5/ The biggest mistake:\n\n"
            f"Treating {topic} as a one-time setup.\n\n"
            f"It needs:\n"
            f"→ Continuous monitoring\n"
            f"→ Regular updates\n"
            f"→ Adapting to new patterns\n\n"
            f"Set it and forget it = expensive technical debt.",

            f"6/ Pro tip most guides skip:\n\n"
            f"Before writing any {topic} code, document:\n"
            f"• What success looks like\n"
            f"• How you'll measure it\n"
            f"• Your rollback plan\n\n"
            f"20-min investment saves 20 hours of debugging.",

            f"7/ When {topic} is done right:\n\n"
            f"📈 Performance improves significantly\n"
            f"💰 Costs reduce over time\n"
            f"🚀 Team velocity increases\n"
            f"😌 Fewer on-call incidents\n\n"
            f"Worth every hour of the investment.",

            f"8/ Want the complete guide with examples?\n\n"
            f"Full article 👇\n{url}\n\n"
            f"If this helped, RT tweet 1 so more devs see it 🙏\n\n{hashtags}",
        ]

        return [EnhancedTweetGenerator._trim_tweet(t) for t in tweets]

    @staticmethod
    def create_poll(post) -> Dict:
        """
        Generate poll data. Polls get 2-3x more engagement.
        Note: Use Twitter API v2 with poll fields to post.
        """
        topic = EnhancedTweetGenerator._extract_topic(post)
        template = random.choice(POLL_TEMPLATES)
        question = template["question"].replace("{topic}", topic)
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"

        tweet_text = (
            f"📊 Quick poll for the dev community:\n\n"
            f"{question}\n\n"
            f"Full breakdown 👇\n{url}\n\n{hashtags}"
        )

        return {
            "text": EnhancedTweetGenerator._trim_tweet(tweet_text),
            "poll_options": template["options"],
            "duration_minutes": 1440,
        }

    @staticmethod
    def create_multiple_variations(post, count: int = 6) -> List[Dict]:
        strategies = ["viral_hook", "thread_preview", "poll_bait",
                      "stat_shock", "list_tease", "hot_take"]
        results = []
        for s in strategies[:count]:
            tweet = EnhancedTweetGenerator.create_engaging_tweet(post, s)
            results.append({
                "strategy": s,
                "tweet": tweet,
                "length": len(tweet),
                "impression_multiplier": EnhancedTweetGenerator._impression_multiplier(s),
            })
        return results

    @staticmethod
    def get_optimal_posting_times() -> List[Dict]:
        """Evidence-based optimal times for tech Twitter (EAT timezone)."""
        return [
            {"time": "07:00", "reason": "Morning commute check — high visibility before work", "priority": 1},
            {"time": "09:30", "reason": "Post-standup scroll — devs checking feeds", "priority": 2},
            {"time": "12:30", "reason": "Lunch break — peak engagement window", "priority": 1},
            {"time": "17:00", "reason": "End-of-day wind-down — high retweet rate", "priority": 2},
            {"time": "20:00", "reason": "Evening learning time — best for threads", "priority": 1},
            {"time": "21:30", "reason": "US East Coast afternoon overlap — global reach", "priority": 3},
        ]

    @staticmethod
    def analyze_tweet_quality(tweet: str) -> Dict:
        score = 0
        feedback = []

        emoji_count = sum(1 for c in tweet if ord(c) > 127000)
        if emoji_count >= 3:
            score += 15
            feedback.append("✅ Strong emoji usage")
        elif emoji_count >= 1:
            score += 8
            feedback.append("⚠️ Consider more emojis")
        else:
            feedback.append("❌ No emojis — add 2-3")

        if "👇" in tweet or "🧵" in tweet:
            score += 20
            feedback.append("✅ Strong CTA / thread signal")
        elif any(w in tweet.lower() for w in ["read", "guide", "full", "here"]):
            score += 10
            feedback.append("⚠️ Weak CTA — add 👇 or 🧵")
        else:
            feedback.append("❌ Missing CTA")

        if "https://" in tweet:
            score += 15
            feedback.append("✅ Link included")
        else:
            feedback.append("❌ No link — add your blog URL")

        hashtag_count = tweet.count("#")
        if 2 <= hashtag_count <= 5:
            score += 10
            feedback.append("✅ Optimal hashtag count (2-5)")
        elif hashtag_count == 1:
            score += 5
            feedback.append("⚠️ Add 1-2 more hashtags")
        elif hashtag_count > 5:
            score += 5
            feedback.append("⚠️ Too many hashtags — cut to 3-5")
        else:
            feedback.append("❌ No hashtags")

        if 100 <= len(tweet) <= 260:
            score += 20
            feedback.append("✅ Optimal length (100-260 chars)")
        elif len(tweet) < 100:
            feedback.append("⚠️ Too short — add more value")
        elif len(tweet) <= 280:
            score += 10
            feedback.append("⚠️ Near limit — trim slightly")
        else:
            feedback.append("❌ Over 280 chars — must trim")

        hook_words = ["unpopular", "secret", "nobody", "stop", "wrong", "mistake",
                      "surprising", "shocking", "finally", "truth", "hidden", "hot take"]
        if any(w in tweet.lower() for w in hook_words):
            score += 15
            feedback.append("✅ Viral hook word detected")
        else:
            feedback.append("⚠️ Add a pattern-interrupt word")

        if any(w in tweet for w in ["?", "reply", "comment", "thoughts"]):
            score += 5
            feedback.append("✅ Invites engagement")

        grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 45 else "D"
        return {"score": score, "grade": grade, "feedback": feedback, "length": len(tweet)}

    # ─────────────────────────────────────────────────────────────
    # STRATEGY IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _viral_hook_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        hook = random.choice(TIER1_HOOKS).replace("{topic}", topic)
        cta = random.choice(CTAS)
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        tweet = f"{hook}\n\n{post.title}\n\n{cta}\n{url}\n\n{hashtags}"
        return EnhancedTweetGenerator._trim_tweet(tweet)

    @staticmethod
    def _thread_preview_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        tweet = (
            f"🧵 THREAD: {post.title}\n\n"
            f"Breaking down {topic} from scratch:\n\n"
            f"→ Core concepts\n"
            f"→ Common pitfalls\n"
            f"→ Best practices\n"
            f"→ Real examples\n\n"
            f"Full guide 👇\n{url}\n\n{hashtags}"
        )
        return EnhancedTweetGenerator._trim_tweet(tweet)

    @staticmethod
    def _poll_bait_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        questions = [
            f"Hot take: most developers overcomplicate {topic}. Agree?",
            f"Be honest — how confident are you with {topic}?",
            f"What's harder: learning {topic} or explaining it to your team?",
        ]
        tweet = (
            f"🗳️ {random.choice(questions)}\n\n"
            f"Drop your answer below 👇\n\n"
            f"Full breakdown: {url}\n\n{hashtags}"
        )
        return EnhancedTweetGenerator._trim_tweet(tweet)

    @staticmethod
    def _stat_shock_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        stats = [
            f"📊 87% of {topic} implementations fail in year one.\n\nThe reason is always the same.",
            f"📈 Companies investing in {topic} see 3x faster growth.\n\nMost still haven't started.",
            f"⏱️ Average developer spends 6 hrs/week on preventable {topic} issues.\n\nThat's 300+ hours/year.",
            f"💸 Poor {topic} architecture costs teams $50k+ in rework.\n\nAvoidable with the right approach.",
        ]
        tweet = (
            f"{random.choice(stats)}\n\n"
            f"How to be in the successful minority 👇\n{url}\n\n{hashtags}"
        )
        return EnhancedTweetGenerator._trim_tweet(tweet)

    @staticmethod
    def _list_tease_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        num = random.choice([5, 7, 9, 10])
        tweet = (
            f"⚡ {num} {topic} things that change how you code:\n\n"
            f"1. [The one most devs skip]\n"
            f"2. [The counterintuitive one]\n"
            f"3. [The time-saver]\n"
            f"...\n\n"
            f"Full list with explanations 👇\n{url}\n\n{hashtags}"
        )
        return EnhancedTweetGenerator._trim_tweet(tweet)

    @staticmethod
    def _question_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        openings = [
            f"What's the hardest part of {topic} nobody warns you about?",
            f"Why does {topic} confuse so many developers?",
            f"What's one {topic} lesson that took you way too long to learn?",
        ]
        tweet = (
            f"💬 {random.choice(openings)}\n\n"
            f"Full research here 👇\n{url}\n\n"
            f"Best answers get shared 🔁\n\n{hashtags}"
        )
        return EnhancedTweetGenerator._trim_tweet(tweet)

    @staticmethod
    def _hot_take_tweet(post) -> str:
        topic = EnhancedTweetGenerator._extract_topic(post)
        url = f"{EnhancedTweetGenerator.BASE_URL}/{post.slug}"
        hashtags = EnhancedTweetGenerator._format_hashtags(post)
        takes = [
            f"Hot take: {topic} is 10% technical, 90% communication.\n\nFight me.",
            f"Controversial: Most {topic} tutorials are teaching it wrong.\n\nHere's what actually works.",
            f"Unpopular opinion: {topic} certifications mean almost nothing without project experience.",
            f"The {topic} skill everyone's chasing in 2026 isn't what you think.",
        ]
        tweet = (
            f"🔥 {random.choice(takes)}\n\n"
            f"Full breakdown with evidence 👇\n{url}\n\n{hashtags}"
        )
        return EnhancedTweetGenerator._trim_tweet(tweet)

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_topic(post) -> str:
        title = getattr(post, "title", "this technology")
        for sep in [":", " - ", " – "]:
            if sep in title:
                title = title.split(sep)[0]
        return title.strip()

    @staticmethod
    def _format_hashtags(post) -> str:
        if hasattr(post, "twitter_hashtags") and post.twitter_hashtags:
            return post.twitter_hashtags
        if hasattr(post, "tags") and post.tags:
            tags = [t.replace(" ", "").replace("-", "") for t in post.tags[:4] if t]
            return " ".join(f"#{t}" for t in tags if t)
        return "#DevTips #Tech"

    @staticmethod
    def _trim_tweet(tweet: str, max_length: int = 280) -> str:
        if len(tweet) <= max_length:
            return tweet
        if "\n\n#" in tweet:
            tweet = tweet.split("\n\n#")[0]
        if len(tweet) <= max_length:
            return tweet
        lines = tweet.split("\n")
        while len("\n".join(lines)) > max_length and len(lines) > 4:
            lines.pop(len(lines) // 2)
        return "\n".join(lines)[:max_length]

    @staticmethod
    def _impression_multiplier(strategy: str) -> float:
        return {
            "viral_hook":     2.5,
            "thread_preview": 3.5,
            "poll_bait":      3.0,
            "stat_shock":     2.8,
            "hot_take":       4.0,
            "list_tease":     2.2,
            "question":       2.6,
        }.get(strategy, 1.0)


if __name__ == "__main__":
    class MockPost:
        title = "Recommender Systems Design: A Complete Guide"
        slug = "recommender-systems-design"
        meta_description = "Master recommender systems: collaborative filtering, content-based models, and hybrid approaches."
        tags = ["MachineLearning", "AI", "Python", "DataScience"]
        twitter_hashtags = "#MachineLearning #AI #Python #DataScience"

    post = MockPost()
    gen = EnhancedTweetGenerator

    print("=" * 70)
    print("TWEET STRATEGY VARIATIONS")
    print("=" * 70)
    for var in gen.create_multiple_variations(post, 6):
        print(f"\n📱 {var['strategy'].upper()} | {var['length']} chars | {var['impression_multiplier']}x")
        print("-" * 70)
        print(var["tweet"])

    print("\n\nOPTIMAL POSTING TIMES (EAT)")
    print("=" * 70)
    for t in gen.get_optimal_posting_times():
        print(f"  {t['time']} — {t['reason']}")