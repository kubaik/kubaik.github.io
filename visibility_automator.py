"""
Visibility Automator — Single-Tweet Edition
Posts one high-engagement tweet per blog post, optimised for impressions
and click-throughs from the X / Twitter algorithm.

Hook philosophy (updated 2026-04-29)
──────────────────────────────────────
  CRITICAL 2026 ALGORITHM CONTEXT:
  • External links are algorithmically suppressed — X's Grok-powered ranking
    gives near-zero median distribution to tweets containing links, especially
    from non-Premium accounts (confirmed March 2026).
  • Replies are worth 27x a like. Conversation depth is the single biggest
    ranking signal. Hooks MUST invite a reply, not just a read.
  • First 30–60 minutes are decisive. Hooks must stop the scroll instantly.
  • Positive/constructive tone gets wider distribution. Combative framing
    is actively penalled even with high engagement.
  • Grok now does semantic analysis — content quality and originality score
    independently of engagement volume.

  HOOK DESIGN PRINCIPLES:
  • Open with tension, a specific number, a paradox, or an honest admission.
    Never open with a topic name, "Just published", or a product pitch.
  • The link sits at the END — after the hook has already earned engagement.
  • Keep the hook line under 10 words where possible. Short = scroll-stopping.
  • Every template closes with a question or an implied gap — this drives replies.
  • Vary templates by slug hash so the same account doesn't sound robotic.
  • Sound like a developer talking at a meetup, not a content scheduler.

  WHAT WORKS IN 2026 (research-backed):
  • Specific numbers: "cut 40%", "3x faster", "in 15 minutes"
  • Honest admissions: "I got this wrong", "took me 6 months to figure out"
  • Paradoxes: "Less code = more reliability. Here's why:"
  • Contrarian takes backed by data: "Everyone says X. The numbers say Y."
  • Short punchy observations the reader feels in their gut
  • Open-ended question at the end to invite replies

CHANGES (2026-04-29):
  - Complete rewrite of _TWEET_TEMPLATES (16 templates, up from 6).
  - Added LINK-FIRST vs LINK-LAST variants: link-last gets more replies first,
    then the algorithm surfaces the link to a wider audience.
  - Added _REPLY_BAIT_ENDINGS: rotating question appended to each tweet to
    drive the reply signal the algorithm weights at 27x a like.
  - Added _build_linkless_hook(): posts a pure text hook with no link when
    account is non-Premium, then relies on the reply thread for clicks.
    (Enabled via config key "twitter_link_strategy": "hook_only" | "link_in_tweet")
  - Trending hashtag is NOT fetched from the Twitter API.
  - Cache file supports an array of hashtags under "hashtags" key.
  - Only ONE tweet is posted per run: post_single_tweet().
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
    "vibe cod":          "Vibe Coding",
    "agentic":           "Agentic AI",
    "mcp":               "MCP",
    "devsecops":         "DevSecOps",
    "platform engineer": "Platform Engineering",
    "post-quantum":      "Post-Quantum Crypto",
    "sovereign":         "Sovereign Cloud",
    "multi-agent":       "Multi-Agent Systems",
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
# ─────────────────────────────────────────────────────────────────

def _load_trending_cache() -> Optional[str]:
    if not _TRENDING_CACHE_FILE.exists():
        print(f"ℹ️  No trending cache file found at {_TRENDING_CACHE_FILE}.")
        return None
    try:
        with open(_TRENDING_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
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
                    f"randomly selected: {chosen} [reference only, not in tweet]"
                )
                return chosen
            print("ℹ️  'hashtags' array in cache is empty.")
            return None
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
    try:
        if isinstance(hashtags, str):
            hashtags = [hashtags]
        with open(_TRENDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"date": datetime.date.today().isoformat(), "hashtags": hashtags},
                f, indent=4,
            )
        print(f"💾 Trending hashtag pool cached: {hashtags}")
    except Exception as e:
        print(f"⚠️  Could not write trending cache: {e}")


def fetch_daily_trending_hashtag(twitter_client) -> Optional[str]:
    """KEPT FOR REFERENCE — no longer called automatically."""
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
# Hashtag resolver
# ─────────────────────────────────────────────────────────────────

def _get_hashtags_for_post(post) -> str:
    if (
        hasattr(post, "twitter_hashtags")
        and post.twitter_hashtags
        and post.twitter_hashtags.strip()
    ):
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
# Reply-bait endings — appended to drive the reply signal
# (replies weighted 27x a like in 2026 X algorithm)
# Each is a genuine question that invites a developer to share experience.
# ─────────────────────────────────────────────────────────────────

_REPLY_BAIT_ENDINGS = [
    "\n\nWhat's the biggest mistake you made here?",
    "\n\nDone this differently? Tell me what worked.",
    "\n\nWhich part took you the longest to get right?",
    "\n\nAnyone hit a different failure mode? Reply below.",
    "\n\nWhat would you add to this?",
    "\n\nWhere do you think most teams still get this wrong?",
    "\n\nHot take: most people skip step 3. Agree?",
    "\n\nWhat's the tool you wish you'd found earlier?",
]


def _pick_reply_bait(slug: str) -> str:
    idx = int(hash(slug + "rb")) % len(_REPLY_BAIT_ENDINGS)
    return _REPLY_BAIT_ENDINGS[idx]


# ─────────────────────────────────────────────────────────────────
# Tweet hook templates — 16 patterns, research-backed for 2026
#
# Design rules applied here:
#  1. Hook line ≤ 10 words. Creates instant scroll-stop.
#  2. Specific number or concrete detail in the first 7 words when possible.
#  3. Link goes at the END — not in line 1 or 2.
#  4. No "Just published", "New post", or topic-name-as-opener.
#  5. Positive/constructive framing (Grok penalises combative tone).
#  6. Each template closes with _REPLY_BAIT_ENDINGS to drive replies.
#
# Placeholders:
#   {topic}  — extracted topic phrase, e.g. "Redis", "System Design"
#   {teaser} — first 60 chars of meta_description
#   {url}    — full post URL
#   {tags}   — hashtag string
#   {bait}   — reply-inviting question
# ─────────────────────────────────────────────────────────────────

_TWEET_TEMPLATES = [
    # 0 — HONEST ADMISSION (high parasocial connection + reply signal)
    # Pattern: "I got [topic] wrong for [time]. Here's what actually works:"
    "I got {topic} wrong for months.\n\n{teaser}\n\nFull breakdown 👇\n{url}\n\n{tags}{bait}",

    # 1 — SPECIFIC NUMBER / RESULT LEAD
    # Pattern: Lead with a concrete outcome, then explain how.
    "{teaser}\n\nHow this works in practice — with real numbers 👇\n{url}\n\n{tags}{bait}",

    # 2 — PARADOX / COUNTER-INTUITIVE OBSERVATION
    # Pattern: Logical tension the reader needs to resolve.
    "Less {topic} code = more reliability.\n\n{teaser}\n\nHere's why 👇\n{url}\n\n{tags}{bait}",

    # 3 — "EVERYONE SAYS X, BUT ACTUALLY Y" (contrarian + data)
    # Pattern proven to get 2M+ impressions in research.
    "Everyone says {topic} is hard.\n\nThe actual hard part is something else.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",

    # 4 — BEFORE / AFTER WITH NUMBERS
    # Pattern: Time-to-fix or cost reduction framing.
    "Before: days debugging {topic}.\nAfter: a 5-step checklist that catches 90% of issues.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",

    # 5 — SHORT PUNCHY OBSERVATION (single idea, reply magnet)
    # Pattern: One sharp insight that stands alone.
    "The teams who nail {topic} all do one thing differently.\n\n{teaser}\n\nI finally noticed what it is 👇\n{url}\n\n{tags}{bait}",

    # 6 — "I ANALYZED [BIG NUMBER]" (authority + curiosity)
    # Research shows this format gets very high save rates.
    "Looked at dozens of {topic} implementations in production.\n\nSame 3 mistakes in almost all of them.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",

    # 7 — CURIOSITY GAP (dev/prod tension)
    # Pattern: Works fine in dev. Fails silently in prod. Why?
    "{topic} works perfectly in dev.\nProduction is a different story.\n\n{teaser}\n\nWhy this happens — and the fix 👇\n{url}\n\n{tags}{bait}",

    # 8 — CONFESSION + LESSON
    # Pattern: Vulnerability + reversal drives emotional engagement.
    "I shipped broken {topic} code to production twice before I understood why.\n\n{teaser}\n\nThe lesson that finally stuck 👇\n{url}\n\n{tags}{bait}",

    # 9 — UNPOPULAR OPINION (highest reply-to-impression ratio)
    # Pattern: Defensible take backed by experience.
    "Unpopular take: most {topic} guides teach the wrong mental model first.\n\n{teaser}\n\nThe model that actually helps 👇\n{url}\n\n{tags}{bait}",

    # 10 — TOOL DISCOVERY (developers hunt for tools that save time)
    # Pattern: "Found X, it replaces Y entirely."
    "Found a {topic} pattern that replaced 200 lines of code with 20.\n\n{teaser}\n\nSetup + full walkthrough 👇\n{url}\n\n{tags}{bait}",

    # 11 — SENIOR VS JUNIOR FRAMING (aspirational, high share rate)
    "Senior devs approach {topic} completely differently.\n\n{teaser}\n\nThe mental shift that changes how you see it 👇\n{url}\n\n{tags}{bait}",

    # 12 — COST / SCALE FRAMING (high relevance to indie hackers + startups)
    "Cut {topic} overhead by doing less of it.\n\n{teaser}\n\nCounter-intuitive breakdown 👇\n{url}\n\n{tags}{bait}",

    # 13 — "THE THING NOBODY MENTIONS" (curiosity + authority)
    "The {topic} docs are good.\nWhat they don't cover is the part that bites you in prod.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",

    # 14 — RESULT + TIMELINE (specific + credible)
    "Fixed a {topic} issue that had been slowing our API by 40%.\nRoot cause was in a single config line.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",

    # 15 — OPEN QUESTION HOOK (highest reply conversion)
    # Pattern: Start with a question the reader can answer from experience.
    "How long did {topic} take you to actually understand?\n\nFor me: embarrassingly long.\n\n{teaser}\n\nHere's what finally made it click 👇\n{url}\n\n{tags}{bait}",
]


# ─────────────────────────────────────────────────────────────────
# Hook style → template index map
# ─────────────────────────────────────────────────────────────────

_STYLE_MAP = {
    "honest_admission":  0,
    "number_lead":       1,
    "paradox":           2,
    "contrarian":        3,
    "before_after":      4,
    "observation":       5,
    "analyzed":          6,
    "dev_prod_gap":      7,
    "confession":        8,
    "unpopular_opinion": 9,
    "tool_discovery":    10,
    "senior_junior":     11,
    "cost_framing":      12,
    "docs_gap":          13,
    "result_timeline":   14,
    "open_question":     15,
    # Legacy aliases (backward-compat)
    "knowledge_gap":     7,
    "curiosity":         7,
    "pattern_interrupt": 9,
    "specific_number":   1,
    "challenge":         9,
}


def _build_single_tweet(
    post,
    base_url: str,
    hook_style: str = "auto",
) -> str:
    """
    Compose one high-engagement tweet for *post*.

    Hashtags come entirely from the post's own metadata.
    The trending cache tag is NOT used here.

    Link suppression note (2026):
      External links are suppressed by X's algorithm for non-Premium accounts.
      The link is still included — Premium accounts need it for click-through —
      but the hook is designed to earn replies BEFORE the link is noticed,
      so the algorithm surfaces the tweet to a wider audience first.
    """
    if hook_style == "auto" or hook_style not in _STYLE_MAP:
        idx = int(hash(post.slug)) % len(_TWEET_TEMPLATES)
    else:
        idx = _STYLE_MAP[hook_style]

    template = _TWEET_TEMPLATES[idx]
    topic = _extract_topic_phrase(post.title, max_words=3)

    raw_desc = getattr(post, "meta_description", "") or ""
    teaser = raw_desc[:65].rstrip()
    if len(raw_desc) > 65:
        teaser = teaser.rsplit(" ", 1)[0] + "…"

    post_url = f"{base_url}/{post.slug}"
    hashtags = _get_hashtags_for_post(post)
    bait = _pick_reply_bait(post.slug)

    tweet = template.format(
        topic=topic, teaser=teaser, url=post_url, tags=hashtags, bait=bait
    )

    # Trim to 280 chars — reduce teaser first, then strip bait if needed
    if len(tweet) > 280:
        budget = 280 - len(tweet) + len(teaser)
        teaser = teaser[:max(budget - 3, 20)].rsplit(" ", 1)[0] + "…"
        tweet = template.format(
            topic=topic, teaser=teaser, url=post_url, tags=hashtags, bait=bait
        )
    if len(tweet) > 280:
        # Last resort: drop the reply-bait ending
        tweet = template.format(
            topic=topic, teaser=teaser, url=post_url, tags=hashtags, bait=""
        )
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet


# ─────────────────────────────────────────────────────────────────
# Link-strategy helpers
# ─────────────────────────────────────────────────────────────────

def _build_linkless_hook(post, hook_style: str = "auto") -> str:
    """
    Build a text-only hook with NO external link.

    Use this for non-Premium accounts where the X algorithm suppresses
    link-containing tweets to near-zero median reach (confirmed March 2026).

    Strategy: post the hook without a link to let the algorithm distribute
    it based on reply/engagement signals. Then drop the URL in the first
    reply to the original tweet.

    Enabled via config key: "twitter_link_strategy": "hook_only"
    """
    if hook_style == "auto" or hook_style not in _STYLE_MAP:
        idx = int(hash(post.slug + "nk")) % len(_TWEET_TEMPLATES)
    else:
        idx = _STYLE_MAP[hook_style]

    template = _TWEET_TEMPLATES[idx]
    topic = _extract_topic_phrase(post.title, max_words=3)

    raw_desc = getattr(post, "meta_description", "") or ""
    teaser = raw_desc[:80].rstrip()
    if len(raw_desc) > 80:
        teaser = teaser.rsplit(" ", 1)[0] + "…"

    hashtags = _get_hashtags_for_post(post)
    bait = _pick_reply_bait(post.slug)

    # Substitute a thread indicator instead of a URL
    tweet = template.format(
        topic=topic, teaser=teaser,
        url="(full guide in the thread below ↓)",
        tags=hashtags, bait=bait,
    )

    if len(tweet) > 280:
        budget = 280 - len(tweet) + len(teaser)
        teaser = teaser[:max(budget - 3, 20)].rsplit(" ", 1)[0] + "…"
        tweet = template.format(
            topic=topic, teaser=teaser,
            url="(full guide in the thread below ↓)",
            tags=hashtags, bait=bait,
        )
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

        self._init_twitter()

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

    # ── Trending hashtag accessor ──────────────────────────────────

    def _get_trending_tag(self) -> Optional[str]:
        return self._trending_tag

    # ── Single-tweet posting (primary method) ─────────────────────

    def post_single_tweet(self, post) -> Dict:
        """
        Compose and post ONE tweet for *post*.

        Respects "twitter_link_strategy" config key:
          "link_in_tweet" (default): include link in tweet body
          "hook_only": post text-only hook; URL goes in first reply

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
        link_strategy = self.config.get(
            "twitter_link_strategy", "link_in_tweet")

        if link_strategy == "hook_only":
            tweet_text = _build_linkless_hook(post, hook_style)
        else:
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

            # If hook_only strategy, post the link as first reply
            if link_strategy == "hook_only":
                post_url = f"{base_url}/{post.slug}"
                reply_text = (
                    f"Full guide here 👇\n{post_url}\n\n"
                    f"(Read time: ~{getattr(post, 'reading_time_minutes', 8)} min)"
                )
                try:
                    self.twitter_client.create_tweet(
                        text=reply_text,
                        in_reply_to_tweet_id=tweet_id,
                    )
                    print(f"📎 Link posted as reply to {tweet_id}")
                except Exception as re_err:
                    print(f"⚠️  Could not post link reply: {re_err}")

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

    # ── Helpers: peak-time awareness ──────────────────────────────

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
            f"Great point on {keyword}. One thing I'd add — consistency in the fundamentals beats chasing every new tool. Wrote a deep dive recently if helpful 👇",
            f"This is spot on. The teams that get {keyword} right share one trait: they treat it as a system, not a checklist. My full breakdown: {{url}}",
            f"Agreed. The biggest mistake with {keyword} is skipping the boring parts early. Costs 10× later. Patterns that actually work: {{url}}",
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

    # ── Existing helpers ──────────────────────────────────────────

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
# CLI entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    print("=" * 70)
    print("VISIBILITY AUTOMATOR — SINGLE TWEET PREVIEW (16 templates)")
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

    print(f"\n🔍 Topic phrase  : '{_extract_topic_phrase(post.title)}'")
    print(f"🏷️  Hashtags       : '{_get_hashtags_for_post(post)}'")
    print(f"💬 Reply bait     : '{_pick_reply_bait(post.slug).strip()}'")
    print(
        f"ℹ️  Cache tag      : '{visibility._get_trending_tag()}' (reference only)")

    print("\n📱 ALL 16 HOOK TEMPLATES")
    print("=" * 70)
    for i, _ in enumerate(_TWEET_TEMPLATES):
        topic = _extract_topic_phrase(post.title)
        teaser = post.meta_description[:65]
        url = f"{base_url}/{post.slug}"
        tags = _get_hashtags_for_post(post)
        bait = _pick_reply_bait(post.slug)
        rendered = _TWEET_TEMPLATES[i].format(
            topic=topic, teaser=teaser, url=url, tags=tags, bait=bait
        )
        print(
            f"\n--- Template {i}: {list(_STYLE_MAP.keys())[i] if i < len(_STYLE_MAP) else 'extra'} ({len(rendered)} chars) ---\n{rendered}")

    print("\n\n🎯 SELECTED TWEET (auto-rotation by slug hash)")
    print("=" * 70)
    selected = _build_single_tweet(post, base_url, hook_style="auto")
    print(f"\n{selected}\n\n({len(selected)} chars)")

    print("\n\n🔗 HOOK-ONLY VARIANT (no link in body)")
    print("=" * 70)
    linkless = _build_linkless_hook(post, hook_style="auto")
    print(f"\n{linkless}\n\n({len(linkless)} chars)")

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
