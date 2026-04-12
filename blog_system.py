import os
import json
import random
import re
import yaml
import asyncio
import aiohttp
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from blog_post import BlogPost
from monetization_manager import MonetizationManager
from seo_optimizer import SEOOptimizer
from visibility_automator import VisibilityAutomator
from static_site_generator import StaticSiteGenerator
from hashtag_manager import HashtagManager, add_hashtags_to_post


# ─────────────────────────────────────────────────────────────────
# Duplicate-detection helpers
# ─────────────────────────────────────────────────────────────────

# Words that carry no topic signal – ignored when comparing titles.
_STOP_WORDS = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is",
    "are", "with", "how", "your", "my", "our", "its", "on", "at",
    "by", "from", "this", "that", "best", "using", "guide", "complete",
    "introduction", "overview", "tutorial", "tips", "top", "ways",
}

# Two posts are "too similar" when their title Jaccard score >= this value.
DUPLICATE_TITLE_THRESHOLD = 0.5


def _tokenise(text: str) -> set:
    """Lowercase, strip punctuation, remove stop-words."""
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_duplicate_title(new_title: str, existing_titles: List[str],
                        threshold: float = DUPLICATE_TITLE_THRESHOLD) -> tuple:
    """
    Return (is_duplicate, best_match_title, score).
    is_duplicate is True when the new title is too close to any existing title.
    """
    new_tokens = _tokenise(new_title)
    best_score = 0.0
    best_match = ""
    for title in existing_titles:
        score = _jaccard(new_tokens, _tokenise(title))
        if score > best_score:
            best_score = score
            best_match = title
    return best_score >= threshold, best_match, best_score


def _load_existing_titles(docs_dir: Path) -> List[str]:
    """Read every post.json and return a list of existing post titles."""
    titles = []
    if not docs_dir.exists():
        return titles
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name == "static":
            continue
        post_json = post_dir / "post.json"
        if post_json.exists():
            try:
                with open(post_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("title", "")
                if title:
                    titles.append(title)
            except Exception:
                pass
    return titles


# ─────────────────────────────────────────────────────────────────
# BlogSystem
# ─────────────────────────────────────────────────────────────────

class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)

        # API keys
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Log key status at startup so CI logs are easy to diagnose
        self._log_key_status()

        # kept for compatibility with monetization / other modules
        self.api_key = self.groq_key or self.openrouter_key

        # Initialize managers
        self.monetization = MonetizationManager(config)
        self.hashtag_manager = HashtagManager(config)

    def _log_key_status(self):
        print("=== API Key Status ===")
        print(
            f"  Groq:        {'configured' if self.groq_key       else 'NOT SET'}")
        print(
            f"  OpenRouter:  {'configured' if self.openrouter_key else 'NOT SET'}")
        print("======================")

    # ─────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────

    def cleanup_posts(self):
        """Clean up incomplete posts and recover from markdown files."""
        print("Cleaning up posts...")

        if not self.output_dir.exists():
            print("No docs directory found.")
            return

        fixed_count = 0
        removed_count = 0

        for post_dir in self.output_dir.iterdir():
            if not post_dir.is_dir():
                continue

            post_json_path = post_dir / "post.json"
            markdown_path = post_dir / "index.md"

            if not post_json_path.exists() and markdown_path.exists():
                try:
                    print(f"Recovering {post_dir.name}...")
                    post = BlogPost.from_markdown_file(
                        markdown_path, post_dir.name)
                    self.save_post(post)
                    fixed_count += 1
                    print(f"Recovered: {post.title}")
                except Exception as e:
                    print(f"Failed to recover {post_dir.name}: {e}")

            elif not post_json_path.exists() and not markdown_path.exists():
                print(f"Removing empty directory: {post_dir.name}")
                try:
                    post_dir.rmdir()
                    removed_count += 1
                except OSError:
                    print(f"Directory not empty: {list(post_dir.iterdir())}")

        print(
            f"Cleanup complete: {fixed_count} recovered, {removed_count} removed")

    # ─────────────────────────────────────────────────────────────
    # API FALLBACK CHAIN: Groq → OpenRouter
    # ─────────────────────────────────────────────────────────────

    async def _call_api_with_fallback(self, messages: List[Dict], max_tokens: int = 3000) -> str:
        """
        1. Try Groq (llama-3.3-70b-versatile).
           - Any error (rate limit or otherwise) -> immediately fall through to OpenRouter.
        2. Try OpenRouter (meta-llama/llama-3.3-70b-instruct:free -- same model quality).
        3. Raise if both fail so the caller triggers the local template fallback.
        """

        # -- 1. Groq ----------------------------------------------
        if self.groq_key:
            try:
                result = await self._call_groq(messages, max_tokens)
                print("API: Groq responded successfully.")
                return result
            except Exception as e:
                print(f"Groq error: {e}")
                print("Falling back to OpenRouter...")
        else:
            print("Groq key not configured — skipping.")

        # ── 2. OpenRouter ─────────────────────────────────────────
        if self.openrouter_key:
            try:
                result = await self._call_openrouter(messages, max_tokens)
                print("API: OpenRouter responded successfully.")
                return result
            except Exception as e:
                print(f"OpenRouter error: {e}")
        else:
            print("OpenRouter key not configured — skipping.")

        raise Exception(
            "All configured API providers failed. "
            "Ensure GROQ_API_KEY and/or OPENROUTER_API_KEY are set as GitHub secrets."
        )

    # ─────────────────────────────────────────────────────────────
    # PROVIDER IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────

    async def _call_groq(self, messages: List[Dict], max_tokens: int) -> str:
        """Call Groq API — llama-3.3-70b-versatile (fast LPU inference)."""
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=data
            ) as response:
                if response.status != 200:
                    raise Exception(f"Groq {response.status}: {await response.text()}")
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def _call_openrouter(self, messages: List[Dict], max_tokens: int) -> str:
        """
        Call OpenRouter — openai/gpt-4o-mini.
        Same 70B model family as Groq, no daily token cap on the free tier.

        Provider routing:
        - Venice is excluded because it rate-limits aggressively on the free tier.
        - allow_fallbacks: true lets OpenRouter try any other available provider
          automatically, so a single upstream 429 never fails the whole request.

        Retries:
        - Connection-level errors (reset by peer, timeout, etc.) are retried up
          to 3 times with a short back-off. These are transient network blips
          common in CI environments and are not API errors.
        - API-level errors (4xx/5xx) are NOT retried — they raise immediately.
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.get("base_url", "https://kubaik.github.io"),
            "X-Title":      self.config.get("site_name", "Tech Blog")
        }
        data = {
            "model": "openai/gpt-4o-mini",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "provider": {
                "ignore": ["Venice"],       # skip the rate-limited upstream
                "allow_fallbacks": True     # try any other provider automatically
            }
        }

        max_attempts = 3
        wait_seconds = [3, 8]   # wait 3s after attempt 1, 8s after attempt 2

        for attempt in range(1, max_attempts + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status != 200:
                            # API error — don't retry, raise immediately
                            raise Exception(f"OpenRouter {response.status}: {await response.text()}")
                        result = await response.json()
                        if "error" in result:
                            raise Exception(
                                f"OpenRouter error payload: {result['error']}")
                        return result["choices"][0]["message"]["content"]

            except aiohttp.ClientConnectionError as e:
                # Connection reset, DNS failure, etc. — worth retrying
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"OpenRouter connection error (attempt {attempt}/{max_attempts}): {e}")
                    print(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"OpenRouter connection failed after {max_attempts} attempts: {e}")

            except asyncio.TimeoutError:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"OpenRouter timeout (attempt {attempt}/{max_attempts}). Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"OpenRouter timed out after {max_attempts} attempts.")

    # ─────────────────────────────────────────────────────────────
    # CONTENT GENERATION
    # ─────────────────────────────────────────────────────────────

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            print("No API keys configured. Using local template content.")
            return self._generate_fallback_post(topic)

        try:
            print(f"Generating content for: {topic}")

            # ── Duplicate-title guard ──────────────────────────────
            existing_titles = _load_existing_titles(self.output_dir)
            title = await self._generate_unique_title(topic, keywords, existing_titles)
            # ──────────────────────────────────────────────────────

            content = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(topic, title)
            slug = self._create_slug(title)

            if not keywords:
                keywords = await self._generate_keywords(topic, title)

            post = BlogPost(
                title=title.strip(),
                content=content.strip(),
                slug=slug,
                tags=keywords[:5],
                meta_description=meta_description.strip(),
                featured_image=f"/static/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=keywords,
                affiliate_links=[],
                monetization_data={}
            )

            # Monetization
            enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
                post.content, topic
            )
            post.content = enhanced_content
            post.affiliate_links = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(
                enhanced_content)

            # Trending hashtags
            print("Generating trending hashtags...")
            hashtags = await self.hashtag_manager.get_daily_hashtags(topic, max_hashtags=10)
            post.tags = list(set(post.tags + hashtags))[:15]
            post.seo_keywords = list(set(post.seo_keywords + hashtags))[:15]
            post.twitter_hashtags = self.hashtag_manager.format_hashtags_for_twitter(
                hashtags[:5])
            print(f"Hashtags: {', '.join(hashtags[:5])}")

            return post

        except Exception as e:
            print(
                f"All API providers exhausted ({e}). Using local template content.")
            return self._generate_fallback_post(topic)

    # ─────────────────────────────────────────────────────────────
    # INDIVIDUAL GENERATION STEPS (each goes through fallback chain)
    # ─────────────────────────────────────────────────────────────

    async def _generate_unique_title(self, topic: str, keywords: List[str],
                                     existing_titles: List[str],
                                     max_attempts: int = 3) -> str:
        """
        Generate a title and keep retrying (up to max_attempts) if the
        result is too similar to an existing post title.
        """
        for attempt in range(1, max_attempts + 1):
            # Ask the model for a fresh title each attempt.
            # On retry we tell it explicitly which title to avoid.
            extra_instruction = ""
            if attempt > 1:
                extra_instruction = (
                    f" IMPORTANT: Do NOT produce a title similar to any of these: "
                    + ", ".join(f'"{t}"' for t in existing_titles[:10])
                    + ". Choose a clearly different angle."
                )

            title = await self._generate_title(topic, keywords,
                                               extra_instruction=extra_instruction)
            title = title.strip().strip('"')

            if not existing_titles:
                # No existing posts — accept immediately.
                return title

            is_dup, match, score = _is_duplicate_title(title, existing_titles)
            if not is_dup:
                print(
                    f"Title accepted (similarity {score:.0%} to nearest existing): {title}")
                return title

            print(
                f"Attempt {attempt}: generated title is too similar "
                f"({score:.0%}) to existing post '{match}'. Retrying…"
            )

        # If all retries are too similar, append the current month/year to
        # distinguish the post rather than giving up entirely.
        print("Warning: could not generate a fully unique title. "
              "Appending date suffix to differentiate.")
        suffix = "..."
        return f"{title}{suffix}"

    async def _generate_title(self, topic: str, keywords: List[str] = None,
                              extra_instruction: str = "") -> str:
        keyword_text = f" Focus on keywords: {', '.join(keywords)}" if keywords else ""
        messages = [
            {"role": "system", "content": "You are a skilled blog title writer. Create engaging, SEO-friendly titles."},
            {"role": "user",   "content": (
                f"Generate a compelling blog post title about '{topic}'.{keyword_text} "
                "The title should be catchy, informative, and under 60 characters."
                f"{extra_instruction}"
            )}
        ]
        title = await self._call_api_with_fallback(messages, max_tokens=100)
        return title.strip().strip('"')

    async def _generate_content(self, title: str, topic: str, keywords: List[str] = None) -> str:
        keyword_text = f"\nKeywords to incorporate naturally: {', '.join(keywords)}" if keywords else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced tech blogger who writes detailed, practical articles. "
                    "Always include specific examples, code snippets, real numbers, and actionable insights. "
                    "Avoid generic statements and filler text."
                )
            },
            {
                "role": "user",
                "content": f"""Write a 4,000-word technical blog post with the title: \"{title}\"

Topic: {topic}{keyword_text}

Requirements:
- Write in Markdown format (##, ###)
- Include 2-3 practical code examples with explanations
- Mention specific tools, platforms, or services by name
- Include real metrics, pricing data, or performance benchmarks where relevant
- Provide concrete use cases with implementation details
- Address common problems with specific solutions
- Write 3,500-4,000 words of substantial content
- Use bullet points and numbered lists where appropriate
- Add a strong conclusion with actionable next steps

Avoid:
- Generic phrases like "crucial aspect", "important technology", or "plays a vital role"
- Vague benefits without specifics
- Template-like structure
- Filler content that doesn't add value

Do not include the main title (# {title}) as it will be added automatically."""
            }
        ]
        content = await self._call_api_with_fallback(messages, max_tokens=3000)
        return content.strip()

    async def _generate_meta_description(self, topic: str, title: str) -> str:
        messages = [
            {"role": "system", "content": "You create SEO-optimized meta descriptions."},
            {"role": "user",   "content": (
                f"Write a compelling meta description (under 160 characters) "
                f"for a blog post titled '{title}' about {topic}."
            )}
        ]
        description = await self._call_api_with_fallback(messages, max_tokens=100)
        return description.strip().strip('"')

    async def _generate_keywords(self, topic: str, title: str) -> List[str]:
        messages = [
            {"role": "system", "content": "You generate relevant SEO keywords."},
            {"role": "user",   "content": (
                f"Generate 8-10 relevant SEO keywords for a blog post titled '{title}' about {topic}. "
                "Return as a comma-separated list."
            )}
        ]
        keywords_text = await self._call_api_with_fallback(messages, max_tokens=150)
        keywords = [k.strip().strip('"') for k in keywords_text.split(',')]
        return [k for k in keywords if k][:10]

    # ─────────────────────────────────────────────────────────────
    # LOCAL FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _generate_fallback_post(self, topic: str) -> BlogPost:
        """Generate a template post when all APIs are unavailable."""
        title = f"Understanding {topic}: A Complete Guide"
        slug = self._create_slug(title)

        content = f"""## Introduction

{topic} is a crucial aspect of modern technology that every developer should understand. In this comprehensive guide, we'll explore the key concepts and best practices.

## What is {topic}?

{topic} represents an important area of technology development that has gained significant traction in recent years. Understanding its core principles is essential for building effective solutions.

## Key Benefits

- **Improved Performance**: {topic} can significantly enhance system performance
- **Better Scalability**: Implementing {topic} helps applications scale more effectively
- **Enhanced User Experience**: Users benefit from the improvements that {topic} brings
- **Cost Effectiveness**: Proper implementation can reduce operational costs

## Best Practices

### 1. Planning and Strategy

Before implementing {topic}, it's important to have a clear strategy and understanding of your requirements.

### 2. Implementation Approach

Take a systematic approach to implementation, starting with the fundamentals and building up complexity gradually.

### 3. Testing and Optimization

Regular testing and optimization ensure that your {topic} implementation continues to perform well.

## Common Challenges

When working with {topic}, developers often encounter several common challenges:

1. **Complexity Management**: Keeping implementations simple and maintainable
2. **Performance Optimization**: Ensuring optimal performance across different scenarios
3. **Integration Issues**: Seamlessly integrating with existing systems

## Conclusion

{topic} is an essential technology for modern development. By following best practices and understanding the core concepts, you can successfully implement solutions that deliver real value.

Remember to stay updated with the latest developments in {topic} as the field continues to evolve rapidly."""

        post = BlogPost(
            title=title,
            content=content,
            slug=slug,
            tags=[topic.replace(' ', '-').lower(),
                  'technology', 'development', 'guide'],
            meta_description=f"A comprehensive guide to {topic} covering key concepts, benefits, and best practices for developers.",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            seo_keywords=[topic.lower(), 'guide', 'tutorial',
                          'best practices'],
            affiliate_links=[],
            monetization_data={}
        )

        enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
            post.content, topic
        )
        post.content = enhanced_content
        post.affiliate_links = affiliate_links
        post.monetization_data = self.monetization.generate_ad_slots(
            enhanced_content)

        return post

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _create_slug(self, title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]

    def save_post(self, post: BlogPost):
        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post.to_dict(), f, indent=2, ensure_ascii=False)

        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")

        print(f"Saved post: {post.title} ({post.slug})")
        if post.affiliate_links:
            print(f"  - {len(post.affiliate_links)} affiliate links added")
        print(
            f"  - {post.monetization_data.get('ad_slots', 0)} ad slots configured")


# ─────────────────────────────────────────────────────────────────
# TOPIC PICKER  (with duplicate-topic guard)
# ─────────────────────────────────────────────────────────────────

def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    print(f"Picking topic from {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file {config_path} not found. Run 'python blog_system.py init' first."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    topics = config.get("content_topics", [])
    if not topics:
        raise ValueError("No content_topics found in config.yaml")

    used = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                used = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            used = []

    available = [t for t in topics if t not in used]
    if not available:
        print("All topics used, resetting...")
        available = topics
        used = []

    # ── Duplicate-topic guard ──────────────────────────────────────
    # Load existing post titles once and skip any candidate topic
    # that is already too similar to what has been published.
    docs_dir = Path("./docs")
    existing_titles = _load_existing_titles(docs_dir)

    if existing_titles:
        safe_available = []
        skipped = []
        for candidate in available:
            # Treat the raw topic string as a pseudo-title for comparison
            is_dup, match, score = _is_duplicate_title(
                candidate, existing_titles, threshold=DUPLICATE_TITLE_THRESHOLD
            )
            if is_dup:
                skipped.append((candidate, match, score))
            else:
                safe_available.append(candidate)

        if skipped:
            print(f"Skipped {len(skipped)} topic(s) already covered:")
            for topic, match, score in skipped:
                print(f"  '{topic}' ≈ '{match}' ({score:.0%})")

        if safe_available:
            available = safe_available
        else:
            # All remaining topics have been covered — reset and pick any
            print("All available topics are already covered. Resetting topic history.")
            available = topics
            used = []
    # ──────────────────────────────────────────────────────────────

    topic = random.choice(available)
    used.append(topic)

    with open(history_file, "w") as f:
        json.dump(used, f, indent=2)

    print(f"Selected topic: {topic}")
    return topic


# ─────────────────────────────────────────────────────────────────
# CONFIG INITIALISER
# ─────────────────────────────────────────────────────────────────

def create_sample_config():
    """Create a sample config.yaml file with monetization settings."""
    config = {
        "site_name":        "Tech Blog",
        "site_description": "Cutting-edge insights into technology, AI, and development",
        "base_url":         "https://kubaik.github.io",
        "base_path":        "",

        "amazon_affiliate_tag":        "aiblogcontent-20",
        "google_analytics_id":         "G-DST4PJYK6V",
        "google_adsense_id":           "ca-pub-4477679588953789",
        "google_search_console_key":   "AIzaSyBqIII5-K2quNev9w7iJoH5U4uqIqKDkEQ",
        "google_adsense_verification": "ca-pub-4477679588953789",

        "social_accounts": {
            "twitter":  "@KubaiKevin",
            "linkedin": "your-linkedin-page",
            "facebook": "your-facebook-page"
        },

        "content_topics": [

            # ══════════════════════════════════════════════════════════════════
            # TIER 1 — HIGHEST VIRALITY (broad audience, developers + non-devs)
            # These are the topics that get shared outside the tech bubble.
            # ══════════════════════════════════════════════════════════════════

            # AI FOR EVERYONE (non-developer angle — massive Twitter audience)
            "How AI Is Changing Everyday Life in 2026",
            "ChatGPT vs Claude vs Gemini: Which AI Actually Wins",
            "10 AI Tools That Replace Expensive Software",
            "AI Prompting Secrets Most People Never Learn",
            "How to Use AI to Make Money Online",
            "AI Tools for Students: Study Smarter Not Harder",
            "Free AI Tools That Professionals Actually Use",
            "How Companies Are Using AI to Cut Costs",
            "AI-Generated Content: What's Real and What's Fake",
            "The AI Skills That Will Get You Hired in 2026",
            "How to Build an AI-Powered Side Hustle",
            "AI vs Human: Where Machines Still Fail",
            "The Hidden Dangers of Relying on AI",
            "How Hospitals Are Using AI to Save Lives",
            "AI in Education: The Future of Learning",

            # TECH MONEY & CAREERS (high engagement — salary, jobs, income)
            "Tech Salaries in 2026: Who Earns What",
            "How to Get a $150K Tech Job Without a Degree",
            "Freelance Developer Income: Realistic Numbers",
            "The Tech Skills That Pay the Most Right Now",
            "How to Negotiate a Tech Salary (Scripts That Work)",
            "Remote Tech Jobs: Where to Find Them in 2026",
            "Tech Career Roadmap: From Zero to Employed in 12 Months",
            "Why Senior Developers Leave Big Tech Companies",
            "The Highest-Paying Programming Languages in 2026",
            "How to Build Passive Income as a Developer",
            "Breaking Into Tech at 30, 40, or 50",
            "Tech Interview Red Flags That Cost Candidates Jobs",
            "The Fastest Growing Tech Roles Right Now",
            "Why Developers Burn Out and How to Prevent It",
            "How to Get Promoted Faster in Tech",

            # STARTUPS & BUILDERS (entrepreneurs + developers both engage)
            "How to Build a SaaS Product as a Solo Developer",
            "The Tech Stack for Bootstrapped Startups in 2026",
            "How Indie Hackers Are Making $10K/Month",
            "No-Code vs Code: When to Use Each",
            "From Idea to Launch: Building an MVP in 30 Days",
            "How to Validate a Startup Idea Before Building",
            "The Cheapest Way to Deploy a Web App in 2026",
            "Why Most Side Projects Fail (And How to Fix It)",
            "How to Find Your First 100 Customers as a Developer",
            "Building in Public: What Works and What Doesn't",
            "Open Source Projects That Made Millions",
            "How to Price Your SaaS Product",
            "What VCs Actually Look for in Tech Startups",
            "Startup Failure Lessons from Founders Who Lost Everything",
            "The Solo Developer's Guide to Scaling",

            # AI TOOLS & PRODUCTIVITY (everyone wants to be more productive)
            "The AI Workflow That Saves 10 Hours a Week",
            "Best AI Coding Assistants Compared: Copilot vs Cursor vs Others",
            "How to Use AI for Content Creation (Without It Sounding Robotic)",
            "AI for Data Analysis: No Coding Required",
            "The Best AI Image Tools in 2026",
            "How Developers Are Using AI to 10x Their Output",
            "Automating Your Life with AI and Python",
            "AI Tools That Write Better Code Than Most Juniors",
            "How to Build a Personal AI Assistant",
            "The Prompt Engineering Techniques That Actually Work",
            "AI Agents: What They Are and Why They Matter",
            "How to Use AI to Learn Any Skill Faster",
            "The Best Free AI APIs for Developers",
            "Building AI-Powered Apps Without Machine Learning Knowledge",
            "AI for Personal Finance: Tools and Strategies",

            # FUTURE OF WORK & TECH SOCIETY (widely shared, opinion-generating)
            "Will AI Replace Software Developers? The Honest Answer",
            "Remote Work vs Office: What the Data Actually Shows",
            "The 4-Day Work Week in Tech: Companies Trying It",
            "Why So Many Developers Are Leaving Big Tech",
            "Tech Layoffs 2026: What's Really Happening",
            "The Skills That Will Be Worthless in 5 Years",
            "How Social Media Algorithms Work (And How to Beat Them)",
            "The Dark Side of Big Tech: What Insiders Say",
            "Why Everyone Should Learn to Code (And Why That's Wrong)",
            "How AI Is Making Wealth Inequality Worse",
            "The Countries Winning the AI Race",
            "Digital Nomad Life: The Real Costs Nobody Mentions",
            "Why Junior Developer Jobs Are Disappearing",
            "The Tech Industry's Mental Health Crisis",
            "How Technology Is Changing Human Relationships",

            # TECH BUSINESS & STRATEGY (decision-makers and entrepreneurs engage)
            "Why Most Digital Transformations Fail",
            "How Netflix Decides What to Build Next",
            "The Tech Behind Amazon's One-Click Empire",
            "How Stripe Became the Internet's Payment System",
            "Why WhatsApp Was Worth $19 Billion",
            "The Engineering Culture That Built Google",
            "How Apple Maintains Premium Pricing in a Competitive Market",
            "The Lessons Startups Learn Too Late",
            "How Big Tech Makes Money From Your Data",
            "The Open Source Business Models That Work",
            "Why Slack Failed to Beat Microsoft Teams",
            "How Figma Was Built and Why Adobe Tried to Buy It",
            "The API Economy: How Twilio and Stripe Print Money",
            "Why Most Tech Companies Never Become Profitable",
            "Platform Business Models Explained",

            # ══════════════════════════════════════════════════════════════════
            # TIER 2 — STRONG VIRALITY (developer-focused but widely shareable)
            # ══════════════════════════════════════════════════════════════════

            # AI & MACHINE LEARNING (keep the best, cut the ultra-academic)
            "Generative AI and Large Language Models Explained",
            "How Neural Networks Actually Learn",
            "Prompt Engineering for Real-World Applications",
            "Vector Databases and Embeddings: A Practical Guide",
            "Building AI Agents That Actually Work",
            "MLOps: Deploying AI Models Without Breaking Everything",
            "Fine-Tuning LLMs Without a GPU",
            "AI Model Monitoring: Catching Drift Before It Hurts",
            "Retrieval-Augmented Generation (RAG) Explained Simply",
            "Multi-Modal AI: When Models See, Hear, and Read",
            "AI Ethics: The Problems Big Tech Doesn't Want to Discuss",
            "Federated Learning: AI That Respects Privacy",
            "AI for Time Series Forecasting in Practice",
            "Explainable AI: Making Black Boxes Transparent",
            "Computer Vision Applications in the Real World",

            # CYBERSECURITY (strong public interest after every breach)
            "How Hackers Actually Break Into Systems",
            "The Biggest Data Breaches of 2026 and What Went Wrong",
            "Zero Trust Security: Why Perimeter Defense Is Dead",
            "How to Protect Your Personal Data from Corporations",
            "Password Security: What Actually Works in 2026",
            "How Ransomware Attacks Work and How to Survive One",
            "API Security Mistakes That Got Companies Hacked",
            "OAuth 2.0 and JWT Authentication Deep Dive",
            "Penetration Testing: How Ethical Hackers Think",
            "Social Engineering: The Human Side of Cybersecurity",
            "Cloud Security Best Practices for Developers",
            "How to Run a Security Audit on Your Web App",
            "DDoS Attacks: How They Work and How to Stop Them",
            "Container Security in Production Kubernetes",
            "The Security Vulnerabilities in Most Mobile Apps",

            # WEB DEVELOPMENT (keep practical, cut framework reference docs)
            "React vs Next.js vs Remix: Choosing the Right Tool",
            "Full-Stack Development in 2026: The Best Stack",
            "Building Real-Time Apps with WebSockets",
            "Web Performance: Why Your Site Loads Slow and How to Fix It",
            "TypeScript Patterns That Actually Save Time",
            "CSS Tricks That Senior Developers Actually Use",
            "Building a Production-Ready App with Supabase",
            "The Jamstack in 2026: Still Worth It?",
            "Web Accessibility: The Laws and the Code",
            "GraphQL vs REST vs tRPC in 2026",
            "Progressive Web Apps: When They Beat Native",
            "Server Components vs Client Components Explained",
            "WebAssembly: When JavaScript Isn't Fast Enough",
            "Building a Micro-Frontend Architecture",
            "Modern Authentication Patterns for Web Apps",

            # BACKEND & SYSTEM DESIGN (system design is perpetually viral)
            "System Design Interview: How to Think Like a Senior Engineer",
            "How Netflix Handles 200 Million Concurrent Streams",
            "Designing a URL Shortener That Handles Billions of Requests",
            "Microservices vs Monolith: The Honest Comparison",
            "Event-Driven Architecture in Practice",
            "Database Indexing: The Hidden Performance Secret",
            "Redis in Production: Patterns That Scale",
            "Building an API That Can Handle a Million Requests",
            "The CAP Theorem Explained Simply",
            "How Stripe Processes Payments Without Losing Data",
            "Designing a Chat System Like WhatsApp",
            "Rate Limiting Strategies That Actually Work",
            "Message Queues: Kafka vs RabbitMQ vs SQS",
            "Database Sharding: When and How to Do It",
            "Serverless Architecture: Real Costs and Real Limits",

            # DEVOPS & CLOUD (focus on pain points, not product docs)
            "The DevOps Mistakes That Cause Outages",
            "Kubernetes: When It Helps and When It Hurts",
            "Docker in Production: What No One Tells You",
            "CI/CD Pipelines That Actually Prevent Bugs",
            "Cloud Cost Optimization: Cutting Your AWS Bill in Half",
            "Infrastructure as Code with Terraform: The Real Guide",
            "GitOps: The Deployment Strategy Worth Understanding",
            "Monitoring Your Application Before Users Complain",
            "The On-Call Engineer's Survival Guide",
            "Blue-Green vs Canary Deployments: Which and When",
            "Log Management at Scale: What Works",
            "Site Reliability Engineering Principles That Matter",
            "Multi-Cloud Strategy: Smart or Overkill",
            "Secrets Management: Keeping Credentials Safe",
            "Platform Engineering: Building Internal Developer Platforms",

            # ══════════════════════════════════════════════════════════════════
            # TIER 3 — SOLID SEO (good search traffic, moderate social reach)
            # ══════════════════════════════════════════════════════════════════

            # DATA ENGINEERING & ANALYTICS
            "Building Your First Data Pipeline That Doesn't Break",
            "Apache Kafka for Developers Who Aren't Data Engineers",
            "Data Warehouse vs Data Lake vs Lakehouse: Which One",
            "Real-Time Data Processing at Scale",
            "Data Quality: Why Your Analytics Are Lying to You",
            "Apache Spark Without the Headaches",
            "Snowflake vs BigQuery vs Redshift in 2026",
            "A/B Testing: How to Run Experiments That Mean Something",
            "Data Mesh Architecture Explained",
            "Business Intelligence Tools for Engineering Teams",

            # MOBILE DEVELOPMENT
            "React Native vs Flutter in 2026: Final Answer",
            "Building a Mobile App That Users Don't Delete",
            "Mobile Performance: Why Your App Feels Slow",
            "Push Notifications Done Right",
            "App Store Optimization: What Actually Moves Rankings",
            "Swift for iOS: Patterns That Scale",
            "Kotlin for Android: Modern Development Guide",
            "Mobile Security Vulnerabilities and Fixes",
            "Cross-Platform vs Native: The Real Trade-offs",
            "Mobile CI/CD Automation in Practice",

            # EMERGING TECHNOLOGIES
            "Blockchain Beyond the Hype: Real Use Cases",
            "Web3 Development: What It Actually Takes",
            "IoT Architecture for Developers",
            "Edge Computing: Why It Matters for Your App",
            "Quantum Computing for Software Engineers",
            "AR Development with Apple Vision Pro",
            "Digital Twin Technology in Industry",
            "5G's Real Impact on Application Development",
            "Low-Code Platforms: Threat or Tool for Developers",
            "Robotics Process Automation in Enterprise",

            # SOFTWARE ENGINEERING PRACTICES
            "Clean Code: The Rules That Actually Matter",
            "SOLID Principles Applied to Real Projects",
            "Test-Driven Development That Doesn't Slow You Down",
            "Code Review: How to Give Feedback That Improves Code",
            "Refactoring Legacy Code Without Breaking Everything",
            "Technical Debt: How to Measure and Pay It Down",
            "Design Patterns You'll Actually Use",
            "Documentation That Developers Actually Read",
            "Agile in Practice: What Works, What's Theatre",
            "Pair Programming: When It's Worth It",

            # DEVELOPER MENTAL MODELS & GROWTH
            "How Senior Developers Think Differently",
            "The Mental Models Every Developer Needs",
            "How to Learn a New Programming Language Fast",
            "Debugging Mindset: How Experts Find Bugs",
            "How to Read Other People's Code Effectively",
            "Technical Writing for Developers",
            "Building a Second Brain as a Developer",
            "How to Contribute to Open Source Projects",
            "Developer Productivity: What Research Actually Shows",
            "Managing Up: How Developers Build Influence",

            # PROGRAMMING LANGUAGES (practical, not academic)
            "Python in 2026: What's New and What Changed",
            "JavaScript Features That Changed How We Code",
            "TypeScript Advanced Patterns Worth Learning",
            "Go Concurrency: Why Gophers Love Goroutines",
            "Rust for Developers Coming from Python or JavaScript",
            "Java in 2026: Still Relevant or Time to Move On",
            "Functional Programming Concepts in Practical Code",
            "SQL Tricks That Replace Complex Application Code",
            "Bash Scripting for Developers Who Avoid the Terminal",
            "Python vs Go vs Rust: Choosing for Your Use Case",

            # TOOLS & PRODUCTIVITY (practical, search-friendly)
            "VS Code Setup That Makes You 2x Faster",
            "Git Commands That Senior Developers Use Daily",
            "Terminal Productivity for Developers",
            "Debugging Techniques That Find Bugs in Minutes",
            "API Testing: Beyond Basic Postman Requests",
            "Database Tools Worth Having in 2026",
            "The Developer's Guide to Time Management",
            "Automating Repetitive Dev Tasks with Python",
            "Command Line Tools Every Developer Should Know",
            "Building a Development Environment That Doesn't Frustrate",

            # PERFORMANCE & OPTIMIZATION (good SEO, technical audience)
            "Application Performance Monitoring That Prevents Incidents",
            "Database Query Optimization: Finding and Fixing Slow Queries",
            "Frontend Performance: Core Web Vitals Explained",
            "Image Optimization for Web in 2026",
            "Lazy Loading and Code Splitting in Practice",
            "Memory Leaks: How to Find and Fix Them",
            "Profiling Python Applications for Speed",
            "Network Performance Optimization for APIs",
            "Algorithm Optimization: Practical Big O Analysis",
            "Caching Strategies That Actually Improve Performance",
        ]
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("Created sample config.yaml file with monetization settings")
    print("\nNext steps:")
    print("1. Replace 'your-tag-20' with your Amazon Associates tag")
    print("2. Add your Google Analytics 4 measurement ID")
    print("3. Add your Google AdSense ID (ca-pub-xxxxxxxxxx)")
    print("4. Update social media handles")
    print("5. Add GitHub secrets: GROQ_API_KEY and OPENROUTER_API_KEY")


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "init":
            print("Initializing blog system...")
            create_sample_config()
            os.makedirs("docs/static", exist_ok=True)
            os.makedirs("analytics", exist_ok=True)
            print("Blog system initialized!")
            print(
                "\nAPI chain: Groq (primary) -> OpenRouter (fallback) -> local template")
            print("Add GitHub secrets: GROQ_API_KEY, OPENROUTER_API_KEY")

        elif mode == "auto":
            print("Starting automated blog generation...")
            print("API chain: Groq -> OpenRouter -> local template")

            if not os.path.exists("config.yaml"):
                print("config.yaml not found. Run 'python blog_system.py init' first.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            blog_system = BlogSystem(config)

            try:
                topic = pick_next_topic()
                blog_post = asyncio.run(blog_system.generate_blog_post(topic))
                blog_system.save_post(blog_post)

                generator = StaticSiteGenerator(blog_system)
                generator.generate_site()

                print(f"Post '{blog_post.title}' generated successfully!")

                visibility = VisibilityAutomator(config)

                # ── 1. POST AS THREAD (maximum impressions) ──────────────────
                print("\n🧵 Posting as thread for maximum impressions...")
                thread_result = visibility.post_thread(blog_post)

                if thread_result['success']:
                    print(
                        f"✅ Thread posted ({thread_result['tweet_count']} tweets)")
                    print(f"   First tweet: {thread_result['first_tweet']}")
                    for i, url in enumerate(thread_result['thread_urls'], 1):
                        print(f"   Tweet {i}: {url}")
                else:
                    # Fallback: post single best tweet if thread fails
                    print(
                        f"⚠️  Thread failed ({thread_result.get('error')}). Falling back to single tweet...")
                    single_result = visibility.post_with_best_strategy(
                        blog_post)
                    if single_result['success']:
                        print(f"✅ Single tweet posted: {single_result['url']}")
                    else:
                        print(
                            f"❌ Single tweet also failed: {single_result.get('error')}")

                # ── 2. REPLY TO TRENDING (boost impressions via engagement) ──
                # Note: requires X API Basic tier ($100/mo) for search access.
                # On the free tier this will log a warning and skip gracefully.
                print("\n💬 Replying to trending tech tweets...")
                try:
                    reply_result = visibility.reply_to_trending(
                        post=blog_post,
                        keywords=[
                            "AI tools",
                            "Python tips",
                            "web development",
                            "system design",
                            "machine learning",
                        ],
                        max_replies=3,
                    )
                    if reply_result['success']:
                        print(
                            f"✅ Replied to {reply_result['reply_count']} trending tweet(s)")
                        for r in reply_result['replies_posted']:
                            print(f"   [{r['keyword']}] {r['reply_url']}")
                    else:
                        # Errors are expected on the free X API tier
                        print(
                            f"⚠️  Reply to trending skipped: {reply_result.get('errors', ['unknown error'])}")
                except Exception as reply_err:
                    # Never let reply failures break the overall cron run
                    print(
                        f"⚠️  Reply to trending raised an exception (skipping): {reply_err}")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        elif mode == "build":
            print("Building static site...")

            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("Site rebuilt successfully!")

        elif mode == "cleanup":
            print("Running cleanup...")

            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            blog_system = BlogSystem(config)
            blog_system.cleanup_posts()

            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()
            print("Cleanup and rebuild complete!")

        elif mode == "debug":
            print("Debug mode...")

            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            blog_system = BlogSystem(config)

            print(f"Output directory: {blog_system.output_dir}")
            print(f"Directory exists: {blog_system.output_dir.exists()}")

            if blog_system.output_dir.exists():
                items = list(blog_system.output_dir.iterdir())
                print(f"Items in directory: {len(items)}")
                for item in items:
                    print(
                        f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        post_json = item / "post.json"
                        post_md = item / "index.md"
                        social_json = item / "social_posts.json"
                        print(
                            f"    post.json:         {'Yes' if post_json.exists()   else 'No'}")
                        print(
                            f"    index.md:          {'Yes' if post_md.exists()     else 'No'}")
                        print(
                            f"    social_posts.json: {'Yes' if social_json.exists() else 'No'}")
                        if post_json.exists():
                            try:
                                with open(post_json, 'r') as f:
                                    data = json.load(f)
                                print(
                                    f"    Valid post:        {data.get('title', 'Unknown')}")
                                print(
                                    f"    Affiliate links:   {len(data.get('affiliate_links', []))}")
                                print(
                                    f"    Ad slots:          {data.get('monetization_data', {}).get('ad_slots', 0)}")
                            except Exception as e:
                                print(f"    Invalid JSON: {e}")

            print("\nRunning automatic cleanup...")
            blog_system.cleanup_posts()

            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()

        elif mode == "social":
            print("Generating social media posts for existing content...")

            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            posts = generator._get_all_posts()

            visibility = VisibilityAutomator(config)

            print(f"Generating social media posts for {len(posts)} posts...")
            for post in posts:
                social_posts = visibility.generate_social_posts(post)

                post_dir = blog_system.output_dir / post.slug
                social_file = post_dir / "social_posts.json"
                with open(social_file, 'w', encoding='utf-8') as f:
                    json.dump(social_posts, f, indent=2)

                print(f"Social posts generated for: {post.title}")
                print(f"  Twitter:  {social_posts['twitter'][:50]}...")
                print(f"  LinkedIn: {social_posts['linkedin'][:50]}...")
                print(f"  Reddit:   {social_posts['reddit_title']}")

            print("Done!")

        elif mode == "test-twitter":
            print("Testing Twitter integration...")

            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            visibility = VisibilityAutomator(config)
            connection_test = visibility.test_twitter_connection()
            print(f"Connection test: {connection_test}")

            if connection_test['success']:
                class TestPost:
                    def __init__(self):
                        self.title = "Test - AI Blog System Twitter Integration"
                        self.meta_description = "Testing our automated blog to Twitter posting system."
                        self.slug = "test-twitter-integration"
                        self.tags = ["test", "automation", "blogging"]

                test_post = TestPost()
                social_posts = visibility.generate_social_posts(test_post)
                print(f"\nGenerated Twitter post preview:")
                print(f"  {social_posts['twitter']}")
                print(f"  Length: {len(social_posts['twitter'])} characters")

                response = input("\nPost this test tweet? (y/N): ")
                if response.lower() == 'y':
                    result = visibility.post_to_twitter(test_post)
                    if result['success']:
                        print("Test tweet posted successfully!")
                        print(f"Tweet ID: {result.get('tweet_id')}")
                    else:
                        print(f"Test tweet failed: {result['error']}")
                else:
                    print("Test cancelled.")

        elif mode == "dedup":
            # Convenience: run deduplication directly from blog_system CLI
            print("Running deduplication...")
            import subprocess
            args = ["python", "deduplicate_posts.py",
                    "--delete"] + sys.argv[2:]
            subprocess.run(args)

        else:
            print(
                "Usage: python blog_system.py [init|auto|build|cleanup|debug|social|test-twitter|dedup]")
            print("  init         - Initialize blog system with config")
            print("  auto         - Generate new post and rebuild site")
            print("  build        - Rebuild site")
            print("  cleanup      - Fix missing files and rebuild")
            print("  debug        - Debug current state and rebuild")
            print("  social       - Generate social media posts for existing content")
            print("  test-twitter - Test Twitter API connection")
            print("  dedup        - Remove near-duplicate posts and rebuild site")

    else:
        print("AI Blog System with Monetization")
        print("API chain: Groq (primary) -> OpenRouter (fallback) -> local template")
        print("\nUsage: python blog_system.py [command]")
        print("\nAvailable commands:")
        print("  init         - Initialize blog system with monetization settings")
        print("  auto         - Generate new monetized post and rebuild site")
        print("  build        - Rebuild site with all features")
        print("  cleanup      - Fix posts and rebuild")
        print("  debug        - Analyse current state and rebuild")
        print("  social       - Generate social media posts")
        print("  test-twitter - Test Twitter API connection and posting")
        print("  dedup        - Remove near-duplicate posts and rebuild site")
        print("\nMonetization features:")
        print("  - Automated affiliate link injection")
        print("  - Google AdSense integration with responsive ads")
        print("  - Strategic ad placement slots (header, middle, footer)")
        print("  - SEO optimization with structured data")
        print("  - Social media post generation")
        print("  - RSS feed for subscribers (/rss.xml)")
        print("\nGitHub secrets required (at least one):")
        print("  GROQ_API_KEY       - Primary  (100k tokens/day free, very fast)")
        print("  OPENROUTER_API_KEY - Fallback (same 70B model, no daily token cap)")
