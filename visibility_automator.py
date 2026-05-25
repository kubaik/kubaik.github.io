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

# Verbs that should never appear as the LAST word of a topic phrase inserted
# into a template slot that requires a noun (e.g. "days debugging {topic}").
# If the extracted phrase ends with one of these, we fall back to a safer noun.
_PREDICATE_VERBS = {
    "quit", "leave", "left", "fail", "failed", "break", "broke",
    "die", "died", "slow", "slowed", "work", "worked", "run", "ran",
    "ship", "shipped", "scale", "scaled", "crash", "crashed",
    "burn", "burned", "burnt", "go", "went",
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
    # ── FIX: career/attrition patterns that produce verb-ended phrases ────────
    "senior dev":        "Senior Dev Retention",
    "devs quit":         "Developer Attrition",
    "quit big tech":     "Big Tech Attrition",
    "leave big tech":    "Big Tech Attrition",
    "freelance rate":    "Freelance Rates",
    "freelance dev":     "Freelance Development",
    "calculate":         "Rate Calculation",
}


def _extract_topic_phrase(title: str, max_words: int = 3) -> str:
    """
    Extract a short noun-phrase topic label from a post title.

    Changes vs original:
    - Checks the override table first (unchanged).
    - After building the candidate phrase from meaningful words, validates
      that it does not end with a bare predicate verb (e.g. "quit", "failed").
      If it does, the last word is dropped; if that leaves nothing, fall back
      to the full title truncated to 40 chars.
    - Ensures the result never starts with a lower-case letter when it comes
      from the word-extraction path (capitalises the first word).
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

    # Trim trailing predicate verbs so the phrase reads as a noun topic,
    # not as a sentence fragment (e.g. "Senior Devs Quit" → "Senior Devs").
    candidate = meaningful[:max_words]
    while candidate and candidate[-1].lower() in _PREDICATE_VERBS:
        candidate = candidate[:-1]

    if not candidate:
        # All words were verbs — fall back to raw title snippet
        return title[:40]

    # Capitalise first word if it came through lower-case
    result = " ".join(candidate)
    return result[0].upper() + result[1:] if result else title[:40]


# ─────────────────────────────────────────────────────────────────
# Teaser extraction helper  (FIX: Bug B)
# ─────────────────────────────────────────────────────────────────

def _extract_teaser(meta_description: str, max_chars: int = 120) -> str:
    """
    Produce a clean teaser string from a meta description for use inside tweet
    templates.

    Problems solved vs the old inline slice:
    1. The old code did ``raw_desc[:65]`` which could land mid-word or
       mid-sentence, producing fragments like ``"t."`` or ``"akable: -"``.
    2. We now prefer sentence-boundary trimming over character-boundary
       trimming, so the teaser always reads as a complete thought.
    3. If the description itself starts with a lower-case fragment (the LLM
       occasionally returns descriptions that begin mid-sentence), we advance
       to the first sentence that starts with a capital letter.
    4. A hard minimum length of 20 chars prevents single-word teasers.
    """
    if not meta_description:
        return ""

    text = meta_description.strip()

    # ── 1. Skip any leading fragment that starts lower-case ──────────────────
    # Split on sentence boundaries and find the first sentence that:
    #   a) starts with an uppercase letter or a digit
    #   b) is at least 20 chars long
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences: List[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Accept: starts with uppercase letter, digit, or a quote/dash followed
        # by uppercase — covers "Cut API…", "40% faster…", ""Redis…", etc.
        if re.match(r'^[A-Z0-9"\'"\-\u2018\u201c]', sent) and len(sent) >= 20:
            clean_sentences.append(sent)

    if not clean_sentences:
        # Fall back: use the raw text but strip any leading lower-case fragment
        # by advancing past the first ". " or ", " boundary.
        for sep in ('. ', ', ', ': ', ' — '):
            idx = text.find(sep)
            if 0 < idx < 40:
                candidate = text[idx + len(sep):].strip()
                if candidate and re.match(r'^[A-Z0-9]', candidate):
                    text = candidate
                    break
        clean_sentences = [text]

    # ── 2. Trim the first usable sentence to max_chars ───────────────────────
    first = clean_sentences[0]

    if len(first) <= max_chars:
        teaser = first
    else:
        # Prefer to break at a sentence boundary within the budget
        inner = re.split(r'(?<=[.!?])\s+', first[:max_chars + 30])
        if len(inner) > 1 and len(inner[0]) >= 20:
            teaser = inner[0]
        else:
            # Fall back to word-boundary trim — but NEVER trim mid-word
            trimmed = first[:max_chars]
            # Advance past the last space to avoid ending mid-word
            last_space = trimmed.rfind(' ')
            if last_space > max_chars // 2:
                trimmed = trimmed[:last_space]
            teaser = trimmed.rstrip('.,;: ') + '…'

    # ── 3. Safety: strip any trailing incomplete parenthetical ───────────────
    teaser = re.sub(r'\s*\([^)]*$', '', teaser).rstrip()

    return teaser


# ─────────────────────────────────────────────────────────────────
# Trending hashtag — cache read only (used in tweets as a tag slot)
# ─────────────────────────────────────────────────────────────────

def _load_trending_cache() -> Optional[str]:
    """
    Load the trending hashtag cache and return ONE randomly chosen tag.
    Returns a string like '#MachineLearning', or None if cache is absent/empty.
    """
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
                # Ensure it's a single word (no spaces) with # prefix
                chosen = re.sub(r"[^\w]", "", chosen.lstrip("#"))
                chosen = f"#{chosen}"
                print(
                    f"📦 Trending cache pool ({len(valid)} tags) — "
                    f"randomly selected for first slot: {chosen}"
                )
                return chosen
            print("ℹ️  'hashtags' array in cache is empty.")
            return None
        # Legacy single-tag format
        tag = data.get("hashtag", "").strip()
        if tag:
            tag = re.sub(r"[^\w]", "", tag.lstrip("#"))
            tag = f"#{tag}"
            print(f"📦 Trending tag from cache (legacy) — will go first: {tag}")
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
# Hashtag helpers
# ─────────────────────────────────────────────────────────────────

def _make_single_word_tag(raw: str) -> str:
    """
    Convert any string into a valid single-word hashtag (no spaces).

    Rules:
      - Strip leading '#'
      - CamelCase each word part so meaning is preserved
      - Remove all non-word characters (punctuation, apostrophes, etc.)
      - Ensure the result is non-empty

    Examples:
      "How to pass technical interviews" → "HowToPassTechnicalInterviews"  (too long but valid)
      "MachineLearning"                  → "MachineLearning"               (unchanged)
      "cloud-native"                     → "CloudNative"
      "#DevOps"                          → "DevOps"
    """
    raw = raw.lstrip("#").strip()
    # Split on spaces, hyphens, underscores
    parts = re.split(r"[\s\-_]+", raw)
    tag = "".join(p.capitalize() if p.islower() else p for p in parts if p)
    # Strip any remaining non-word characters
    tag = re.sub(r"[^\w]", "", tag)
    return tag


def _get_hashtags_for_post(post, max_tags: int = 4) -> str:
    seen: set = set()
    clean: List[str] = []

    # ── 1. Prefer post.twitter_hashtags (already curated by blog_system) ──
    if (
        hasattr(post, "twitter_hashtags")
        and post.twitter_hashtags
        and post.twitter_hashtags.strip()
    ):
        for part in post.twitter_hashtags.strip().split():
            word = _make_single_word_tag(part)
            if word and word.lower() not in seen:
                seen.add(word.lower())
                clean.append(f"#{word}")
            if len(clean) >= max_tags:
                break

    # ── 2. Fall back to post.tags ──────────────────────────────────────────
    if not clean and hasattr(post, "tags") and post.tags:
        for t in post.tags:
            if not t:
                continue
            word = _make_single_word_tag(t)
            if len(word) < 2:
                continue
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            clean.append(f"#{word}")
            if len(clean) >= max_tags:
                break

    # ── 3. Fall back to SEO keywords ──────────────────────────────────────
    if not clean and hasattr(post, "seo_keywords") and post.seo_keywords:
        for kw in post.seo_keywords[:10]:
            kw = kw.strip()
            if not kw:
                continue
            word = _make_single_word_tag(kw)
            if not word or word.lower() in seen:
                continue
            seen.add(word.lower())
            clean.append(f"#{word}")
            if len(clean) >= max_tags:
                break

    # ── 4. Last-resort generic tags ───────────────────────────────────────
    if not clean:
        if hasattr(post, "title") and post.title:
            phrase = _extract_topic_phrase(post.title, max_words=2)
            word = _make_single_word_tag(phrase)
            clean = [f"#{word}", "#Programming", "#SoftwareEngineering"]
        else:
            clean = ["#Programming", "#SoftwareEngineering", "#TechBlog"]

    # ── 5. Inject one random trending tag (always first slot) ─────────────
    trending_tag = _load_trending_cache()
    if trending_tag:
        trending_word = _make_single_word_tag(trending_tag)
        trending_formatted = f"#{trending_word}"
        if trending_word.lower() not in seen:
            if len(clean) >= max_tags:
                # Insert at front, drop last tag to stay within budget
                clean = [trending_formatted] + clean[: max_tags - 1]
            else:
                clean = [trending_formatted] + clean
            seen.add(trending_word.lower())
            print(
                f"  🔥 Trending tag injected as first hashtag: {trending_formatted}")
        else:
            print(
                f"  ℹ️  Trending tag {trending_formatted} already present — skipped.")

    return " ".join(clean[:max_tags])


# ─────────────────────────────────────────────────────────────────
# Reply-bait endings
# ─────────────────────────────────────────────────────────────────

_REPLY_BAIT_ENDINGS = [
    "\n\nWhat broke first when you tried this?",
    "\n\nDone this differently? Tell me what worked.",
    "\n\nWhich part took you the longest to figure out?",
    "\n\nAnyone hit a different failure mode? Reply below.",
    "\n\nWhat would you add — or cut — from this list?",
    "\n\nWhere do most teams still get this wrong?",
    "\n\nHot take: the measurement step gets skipped every time. Agree?",
    "\n\nWhat's the tool you wish you'd found six months earlier?",
    "\n\nWhat's your biggest lesson from getting this wrong?",
    "\n\nIf you've shipped this in prod — what surprised you most?",
    "\n\nWhat's the one thing you'd tell your past self before starting this?",
    "\n\nWhere did the docs lead you astray on this one?",
]


def _pick_reply_bait(slug: str) -> str:
    idx = int(hash(slug + "rb")) % len(_REPLY_BAIT_ENDINGS)
    return _REPLY_BAIT_ENDINGS[idx]


# ─────────────────────────────────────────────────────────────────
# Tweet trimming helper  (FIX: Bug C)
# ─────────────────────────────────────────────────────────────────

def _trim_to_budget(text: str, budget: int) -> str:
    """
    Trim *text* to at most *budget* characters while preserving clean grammar.

    Strategy (in order of preference):
    1. No trimming needed — return as-is.
    2. Break at the last sentence-ending punctuation (. ! ?) within budget.
    3. Break at the last em-dash (—) or semicolon within budget (clause boundary).
    4. Break at the last comma within budget (phrase boundary).
    5. Break at the last space within budget (word boundary).
    6. Hard cut at budget — last resort, appends "…".

    In all truncation cases a trailing "…" is appended ONLY when the cut is
    not already at a natural sentence end.
    """
    if len(text) <= budget:
        return text

    # Helper: does the string end at a natural sentence boundary?
    def _is_sentence_end(s: str) -> bool:
        return bool(s) and s[-1] in '.!?'

    window = text[:budget]

    # 1. Sentence boundary
    for punct in ('.', '!', '?'):
        pos = window.rfind(punct)
        if pos >= budget // 2:
            candidate = text[:pos + 1].rstrip()
            if len(candidate) <= budget:
                return candidate

    # 2. Clause boundary (— or ;)
    for sep in ('—', ';'):
        pos = window.rfind(sep)
        if pos >= budget // 2:
            candidate = text[:pos].rstrip().rstrip(',;')
            if candidate:
                return candidate + '…'

    # 3. Comma
    pos = window.rfind(',')
    if pos >= budget // 2:
        candidate = text[:pos].rstrip()
        if candidate:
            return candidate + '…'

    # 4. Word boundary
    pos = window.rfind(' ')
    if pos > 0:
        candidate = text[:pos].rstrip('.,;: ')
        return candidate + '…'

    # 5. Hard cut
    return window.rstrip() + '…'


# ─────────────────────────────────────────────────────────────────
# Tweet hook templates — 16 patterns
# ─────────────────────────────────────────────────────────────────

_TWEET_TEMPLATES = [
    # 0 — HONEST ADMISSION
    "Many developers get {topic} wrong for months.\n\n{teaser}\n\nFull breakdown 👇\n{url}\n\n{tags}{bait}",
    # 1 — SPECIFIC NUMBER / RESULT LEAD
    "{teaser}\n\nHow this works in practice — with real numbers 👇\n{url}\n\n{tags}{bait}",
    # 2 — PARADOX
    "Less {topic} code = more reliability.\n\n{teaser}\n\nHere's why 👇\n{url}\n\n{tags}{bait}",
    # 3 — CONTRARIAN
    "Everyone says {topic} is hard.\n\nThe actual hard part is something else.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",
    # 4 — BEFORE / AFTER
    "Before: days debugging {topic}.\nAfter: a 5-step checklist that catches 90% of issues.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",
    # 5 — OBSERVATION
    "Teams that nail {topic} all do one thing differently.\n\n{teaser}\n\nHere's what sets them apart 👇\n{url}\n\n{tags}{bait}",
    # 6 — ANALYZED
    "Dozens of {topic} implementations reviewed.\n\nSame 3 mistakes in almost all of them.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",
    # 7 — DEV/PROD GAP
    "{topic} works perfectly in dev.\nProduction is a different story.\n\n{teaser}\n\nWhy this happens — and the fix 👇\n{url}\n\n{tags}{bait}",
    # 8 — CONFESSION
    "Most teams ship broken {topic} code to production before they understand why.\n\n{teaser}\n\nThe lesson that finally sticks 👇\n{url}\n\n{tags}{bait}",
    # 9 — UNPOPULAR OPINION
    "Unpopular take: most {topic} guides teach the wrong mental model first.\n\n{teaser}\n\nThe model that actually helps 👇\n{url}\n\n{tags}{bait}",
    # 10 — TOOL DISCOVERY
    "A {topic} pattern that replaces 200 lines of code with 20.\n\n{teaser}\n\nSetup + full walkthrough 👇\n{url}\n\n{tags}{bait}",
    # 11 — SENIOR VS JUNIOR
    "Senior devs approach {topic} completely differently.\n\n{teaser}\n\nThe mental shift that changes how you see it 👇\n{url}\n\n{tags}{bait}",
    # 12 — COST / SCALE
    "Cut {topic} overhead by doing less of it.\n\n{teaser}\n\nCounter-intuitive breakdown 👇\n{url}\n\n{tags}{bait}",
    # 13 — DOCS GAP
    "The {topic} docs are good.\nWhat they don't cover is the part that bites you in prod.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",
    # 14 — RESULT + TIMELINE
    "A {topic} issue was slowing API responses by 40%.\nRoot cause was in a single config line.\n\n{teaser}\n\n👇\n{url}\n\n{tags}{bait}",
    # 15 — OPEN QUESTION
    "How long did {topic} take your team to actually get right?\n\n{teaser}\n\nWhat finally made it click 👇\n{url}\n\n{tags}{bait}",
]

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
    # Legacy aliases
    "knowledge_gap":     7,
    "curiosity":         7,
    "pattern_interrupt": 9,
    "specific_number":   1,
    "challenge":         9,
}

# Templates where {topic} is used as a direct object / noun in context.
# For these we validate the extracted phrase is noun-safe (done in
# _extract_topic_phrase already via _PREDICATE_VERBS), but we also
# provide a graceful inline fallback label per template so the sentence
# always reads correctly even if the topic phrase is short or unusual.
_TOPIC_USES_NOUN_SLOT = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}


def _build_single_tweet(post, base_url: str, hook_style: str = "auto") -> str:
    if hook_style == "auto" or hook_style not in _STYLE_MAP:
        idx = int(hash(post.slug)) % len(_TWEET_TEMPLATES)
    else:
        idx = _STYLE_MAP[hook_style]

    template = _TWEET_TEMPLATES[idx]
    topic = _extract_topic_phrase(post.title, max_words=3)

    # ── FIX Bug B: use the robust teaser extractor ───────────────────────────
    raw_desc = getattr(post, "meta_description", "") or ""
    # Teaser budget: leave room for the rest of the template structure.
    # 120 chars is generous; _build_single_tweet trims the full tweet below.
    teaser = _extract_teaser(raw_desc, max_chars=120)

    post_url = f"{base_url}/{post.slug}"
    reading_time = getattr(post, 'reading_time_minutes', None)
    url_line = f"{post_url}  ({reading_time} min read)" if reading_time else post_url

    hashtags = _get_hashtags_for_post(post)
    bait = _pick_reply_bait(post.slug)

    tweet = template.format(
        topic=topic, teaser=teaser, url=url_line, tags=hashtags, bait=bait
    )

    if len(tweet) > 280:
        # ── FIX Bug B (secondary): re-trim teaser using _trim_to_budget ──────
        # How many chars does the non-teaser skeleton cost?
        skeleton = template.format(
            topic=topic, teaser="", url=url_line, tags=hashtags, bait=bait
        )
        teaser_budget = max(20, 280 - len(skeleton))
        teaser = _trim_to_budget(teaser, teaser_budget)
        tweet = template.format(
            topic=topic, teaser=teaser, url=url_line, tags=hashtags, bait=bait
        )

    if len(tweet) > 280:
        # Drop read-time annotation from URL to recover space
        skeleton = template.format(
            topic=topic, teaser="", url=post_url, tags=hashtags, bait=""
        )
        teaser_budget = max(20, 280 - len(skeleton))
        teaser = _trim_to_budget(teaser, teaser_budget)
        tweet = template.format(
            topic=topic, teaser=teaser, url=post_url, tags=hashtags, bait=""
        )

    if len(tweet) > 280:
        tweet = _trim_to_budget(tweet, 277)

    return tweet


# ─────────────────────────────────────────────────────────────────
# Link-strategy helpers
# ─────────────────────────────────────────────────────────────────

def _build_linkless_hook(post, hook_style: str = "auto") -> str:
    if hook_style == "auto" or hook_style not in _STYLE_MAP:
        idx = int(hash(post.slug + "nk")) % len(_TWEET_TEMPLATES)
    else:
        idx = _STYLE_MAP[hook_style]

    template = _TWEET_TEMPLATES[idx]
    topic = _extract_topic_phrase(post.title, max_words=3)

    # ── FIX Bug B: use the robust teaser extractor ───────────────────────────
    raw_desc = getattr(post, "meta_description", "") or ""
    teaser = _extract_teaser(raw_desc, max_chars=120)

    hashtags = _get_hashtags_for_post(post)
    bait = _pick_reply_bait(post.slug)

    tweet = template.format(
        topic=topic, teaser=teaser,
        url="(full guide in the thread below ↓)",
        tags=hashtags, bait=bait,
    )

    if len(tweet) > 280:
        skeleton = template.format(
            topic=topic, teaser="",
            url="(full guide in the thread below ↓)",
            tags=hashtags, bait=bait,
        )
        teaser_budget = max(20, 280 - len(skeleton))
        teaser = _trim_to_budget(teaser, teaser_budget)
        tweet = template.format(
            topic=topic, teaser=teaser,
            url="(full guide in the thread below ↓)",
            tags=hashtags, bait=bait,
        )

    if len(tweet) > 280:
        tweet = _trim_to_budget(tweet, 277)

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

        # Cache is now consumed by _get_hashtags_for_post() per tweet.
        # We load it once here only for the reference log line.
        self._trending_tag: Optional[str] = _load_trending_cache()
        if self._trending_tag:
            print(
                f"ℹ️  Trending tag available — will be injected into tweet hashtag slot: "
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

    # ── Compose preview (dry-run, no API call) ────────────────────

    def compose_tweet_preview(self, post) -> Dict:
        """
        Compose the tweet that *would* be posted, without calling the API.
        Used as the template fallback when post.prewritten_tweet is empty.
        Always safe to call — no credentials required.

        Returns:
            {
                'tweet_text': str,
                'char_count': int,
                'hook_style': str,
                'link_strategy': str,
            }
        """
        base_url = self.config.get("base_url", "https://kubaik.github.io")
        hook_style = self.config.get("hook_style", "auto")
        link_strategy = self.config.get(
            "twitter_link_strategy", "link_in_tweet")

        if link_strategy == "hook_only":
            tweet_text = _build_linkless_hook(post, hook_style)
        else:
            tweet_text = _build_single_tweet(post, base_url, hook_style)

        return {
            "tweet_text":    tweet_text,
            "char_count":    len(tweet_text),
            "hook_style":    hook_style,
            "link_strategy": link_strategy,
        }

    # ── NEW: Post a prewritten tweet verbatim ─────────────────────

    def post_prewritten_tweet(self, post, tweet_text: str) -> Dict:
        """
        Post *tweet_text* verbatim — no internal template composition.

        Called by blog_system.py auto mode when the LLM wrote the tweet
        during content generation (stored as post.prewritten_tweet).
        This bypasses post_single_tweet()'s internal _build_single_tweet()
        call so the polished, context-aware text is used exactly as written.

        Supports the same hook_only link strategy as post_single_tweet():
        if "twitter_link_strategy" is "hook_only", the URL is posted as a
        first reply after the main tweet so the algorithm distributes the
        link-free hook to a wider audience first.

        Parameters
        ----------
        post : BlogPost
            Used for logging and the hook_only URL reply construction.
        tweet_text : str
            The final tweet text to post — must be ≤ 280 characters.

        Returns
        -------
        dict
            {"success": True,  "tweet_id": str, "url": str, "char_count": int}
            {"success": False, "error": str}
        """
        if not self.twitter_client:
            return {"success": False, "error": "Twitter client not initialized."}

        base_url = self.config.get("base_url", "https://kubaik.github.io")
        link_strategy = self.config.get(
            "twitter_link_strategy", "link_in_tweet")

        SEP = "─" * 68
        print(f"\n{SEP}")
        print("📣  TWITTER POST — PRE-FLIGHT (bundle tweet)")
        print(SEP)
        print(f"  Post title    : {getattr(post, 'title', '(unknown)')}")
        print(f"  Slug          : {getattr(post, 'slug',  '(unknown)')}")
        print(f"  Source        : LLM-generated during content creation")
        print(f"  Link strategy : {link_strategy}")
        print(f"  Char count    : {len(tweet_text)} / 280")

        try:
            response = self.twitter_client.create_tweet(text=tweet_text)
            tweet_id = response.data["id"]
            username = self._username or "i"
            tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"

            # hook_only: post URL as first reply so the algorithm distributes
            # the link-free hook first, then surfaces the link to a wider audience.
            reply_url = None
            if link_strategy == "hook_only":
                post_url = f"{base_url}/{post.slug}"
                reply_text = (
                    f"Full guide here 👇\n{post_url}\n\n"
                    f"(Read time: ~{getattr(post, 'reading_time_minutes', 8)} min)"
                )
                try:
                    reply_resp = self.twitter_client.create_tweet(
                        text=reply_text,
                        in_reply_to_tweet_id=tweet_id,
                    )
                    reply_url = (
                        f"https://twitter.com/{username}/status/"
                        f"{reply_resp.data['id']}"
                    )
                    print(f"  📎  Link reply posted : {reply_url}")
                except Exception as re_err:
                    print(f"  ⚠️  Link reply failed : {re_err}")

            print(SEP)
            print("✅  TWITTER POST — SUCCESS")
            print(SEP)
            print(f"  Tweet ID      : {tweet_id}")
            print(f"  Tweet URL     : {tweet_url}")
            print(f"  Char count    : {len(tweet_text)} / 280")
            if reply_url:
                print(f"  Link reply    : {reply_url}")
            print(SEP + "\n")

            return {
                "success":    True,
                "tweet_id":   tweet_id,
                "url":        tweet_url,
                "tweet_text": tweet_text,
                "char_count": len(tweet_text),
            }

        except Exception as e:
            print("❌  TWITTER POST — FAILED")
            print(f"  Error         : {e}")
            return {"success": False, "error": str(e), "tweet_text": tweet_text}

    # ── Single-tweet posting (template path) ──────────────────────

    def post_single_tweet(self, post) -> Dict:
        """
        Compose and post ONE tweet using the internal template engine.

        This is the original template-based path. It is still used when
        post.prewritten_tweet is empty (e.g. fallback posts, social mode).
        For normal auto-mode runs, post_prewritten_tweet() is called instead.

        Respects "twitter_link_strategy" config key:
          "link_in_tweet" (default): include link in tweet body
          "hook_only": post text-only hook; URL goes in first reply
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

        SEP = "─" * 68
        print(f"\n{SEP}")
        print("📣  TWITTER POST — PRE-FLIGHT (template)")
        print(SEP)
        print(f"  Post title    : {getattr(post, 'title', '(unknown)')}")
        print(f"  Slug          : {getattr(post, 'slug',  '(unknown)')}")
        print(f"  Post URL      : {base_url}/{getattr(post, 'slug', '')}")
        print(f"  Hook style    : {hook_style}")
        print(f"  Link strategy : {link_strategy}")
        print(f"  Char count    : {len(tweet_text)} / 280")
        print(f"  Hashtags      : {_get_hashtags_for_post(post)}")
        print(
            f"  Meta desc     : {getattr(post, 'meta_description', '')[:120]}")
        print(SEP)
        print("📝  TWEET TEXT (full):")
        print(SEP)
        for line in tweet_text.splitlines():
            print(f"  │ {line}")
        print(SEP)

        try:
            response = self.twitter_client.create_tweet(text=tweet_text)
            tweet_id = response.data["id"]
            username = self._username or "i"
            url = f"https://twitter.com/{username}/status/{tweet_id}"

            reply_url = None
            if link_strategy == "hook_only":
                post_url = f"{base_url}/{post.slug}"
                reply_text = (
                    f"Full guide here 👇\n{post_url}\n\n"
                    f"(Read time: ~{getattr(post, 'reading_time_minutes', 8)} min)"
                )
                try:
                    reply_resp = self.twitter_client.create_tweet(
                        text=reply_text,
                        in_reply_to_tweet_id=tweet_id,
                    )
                    reply_url = (
                        f"https://twitter.com/{username}/status/"
                        f"{reply_resp.data['id']}"
                    )
                    print(f"  📎  Link reply posted : {reply_url}")
                except Exception as re_err:
                    print(f"  ⚠️  Link reply failed : {re_err}")

            print(SEP)
            print("✅  TWITTER POST — SUCCESS")
            print(SEP)
            print(f"  Tweet ID      : {tweet_id}")
            print(f"  Tweet URL     : {url}")
            print(f"  Char count    : {len(tweet_text)} / 280")
            if reply_url:
                print(f"  Link reply    : {reply_url}")
            print(SEP + "\n")

            return {
                "success":    True,
                "tweet_id":   tweet_id,
                "url":        url,
                "tweet_text": tweet_text,
                "char_count": len(tweet_text),
            }

        except Exception as e:
            print(SEP)
            print("❌  TWITTER POST — FAILED")
            print(SEP)
            print(f"  Error         : {e}")
            for line in tweet_text.splitlines():
                print(f"  │ {line}")
            print(SEP + "\n")
            return {"success": False, "error": str(e), "tweet_text": tweet_text}

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

    # ── Legacy helpers (kept for backward compatibility) ──────────

    def post_to_twitter(self, tweet_text: str = None, post=None, strategy: str = "auto") -> Dict:
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
        prewritten_tweet = ""  # empty → template path used in preview

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
        f"ℹ️  Cache tag      : '{visibility._get_trending_tag()}' (will be injected into last hashtag slot)")

    print("\n📱 ALL 16 HOOK TEMPLATES")
    print("=" * 70)
    for i, _ in enumerate(_TWEET_TEMPLATES):
        topic = _extract_topic_phrase(post.title)
        teaser = _extract_teaser(post.meta_description, max_chars=120)
        url = f"{base_url}/{post.slug}"
        tags = _get_hashtags_for_post(post)
        bait = _pick_reply_bait(post.slug)
        rendered = _TWEET_TEMPLATES[i].format(
            topic=topic, teaser=teaser, url=url, tags=tags, bait=bait
        )
        print(f"\n--- Template {i}: ({len(rendered)} chars) ---\n{rendered}")

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
