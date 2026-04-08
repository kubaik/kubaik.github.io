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
        self.groq_key       = os.getenv("GROQ_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Log key status at startup so CI logs are easy to diagnose
        self._log_key_status()

        # kept for compatibility with monetization / other modules
        self.api_key = self.groq_key or self.openrouter_key

        # Initialize managers
        self.monetization    = MonetizationManager(config)
        self.hashtag_manager = HashtagManager(config)

    def _log_key_status(self):
        print("=== API Key Status ===")
        print(f"  Groq:        {'configured' if self.groq_key       else 'NOT SET'}")
        print(f"  OpenRouter:  {'configured' if self.openrouter_key else 'NOT SET'}")
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

        fixed_count   = 0
        removed_count = 0

        for post_dir in self.output_dir.iterdir():
            if not post_dir.is_dir():
                continue

            post_json_path = post_dir / "post.json"
            markdown_path  = post_dir / "index.md"

            if not post_json_path.exists() and markdown_path.exists():
                try:
                    print(f"Recovering {post_dir.name}...")
                    post = BlogPost.from_markdown_file(markdown_path, post_dir.name)
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

        print(f"Cleanup complete: {fixed_count} recovered, {removed_count} removed")

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
                            raise Exception(f"OpenRouter error payload: {result['error']}")
                        return result["choices"][0]["message"]["content"]

            except aiohttp.ClientConnectionError as e:
                # Connection reset, DNS failure, etc. — worth retrying
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(f"OpenRouter connection error (attempt {attempt}/{max_attempts}): {e}")
                    print(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(f"OpenRouter connection failed after {max_attempts} attempts: {e}")

            except asyncio.TimeoutError:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(f"OpenRouter timeout (attempt {attempt}/{max_attempts}). Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(f"OpenRouter timed out after {max_attempts} attempts.")

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
            title           = await self._generate_unique_title(topic, keywords, existing_titles)
            # ──────────────────────────────────────────────────────

            content          = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(topic, title)
            slug             = self._create_slug(title)

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
            post.content           = enhanced_content
            post.affiliate_links   = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)

            # Trending hashtags
            print("Generating trending hashtags...")
            hashtags = await self.hashtag_manager.get_daily_hashtags(topic, max_hashtags=10)
            post.tags             = list(set(post.tags + hashtags))[:15]
            post.seo_keywords     = list(set(post.seo_keywords + hashtags))[:15]
            post.twitter_hashtags = self.hashtag_manager.format_hashtags_for_twitter(hashtags[:5])
            print(f"Hashtags: {', '.join(hashtags[:5])}")

            return post

        except Exception as e:
            print(f"All API providers exhausted ({e}). Using local template content.")
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
                print(f"Title accepted (similarity {score:.0%} to nearest existing): {title}")
                return title

            print(
                f"Attempt {attempt}: generated title is too similar "
                f"({score:.0%}) to existing post '{match}'. Retrying…"
            )

        # If all retries are too similar, append the current month/year to
        # distinguish the post rather than giving up entirely.
        print("Warning: could not generate a fully unique title. "
              "Appending date suffix to differentiate.")
        suffix = datetime.now().strftime("%B %Y")
        return f"{title}"

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
        slug  = self._create_slug(title)

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
            tags=[topic.replace(' ', '-').lower(), 'technology', 'development', 'guide'],
            meta_description=f"A comprehensive guide to {topic} covering key concepts, benefits, and best practices for developers.",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            seo_keywords=[topic.lower(), 'guide', 'tutorial', 'best practices'],
            affiliate_links=[],
            monetization_data={}
        )

        enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
            post.content, topic
        )
        post.content           = enhanced_content
        post.affiliate_links   = affiliate_links
        post.monetization_data = self.monetization.generate_ad_slots(enhanced_content)

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
        print(f"  - {post.monetization_data.get('ad_slots', 0)} ad slots configured")


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
        used      = []

    # ── Duplicate-topic guard ──────────────────────────────────────
    # Load existing post titles once and skip any candidate topic
    # that is already too similar to what has been published.
    docs_dir        = Path("./docs")
    existing_titles = _load_existing_titles(docs_dir)

    if existing_titles:
        safe_available = []
        skipped        = []
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
            used      = []
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
            # AI & MACHINE LEARNING
            "Machine Learning Algorithms Explained",
            "Deep Learning Neural Networks",
            "Natural Language Processing Techniques",
            "Computer Vision Applications",
            "Reinforcement Learning Strategies",
            "Generative AI and Large Language Models",
            "AI Model Training Best Practices",
            "Transfer Learning Implementation",
            "AI Ethics and Responsible AI",
            "MLOps and ML Pipeline Automation",
            "Feature Engineering Techniques",
            "Hyperparameter Tuning Methods",
            "AI Model Deployment Strategies",
            "Explainable AI (XAI) Techniques",
            "AutoML and Neural Architecture Search",
            "Time Series Forecasting with AI",
            "Anomaly Detection Using Machine Learning",
            "Recommender Systems Design",
            "AI for Edge Computing",
            "Federated Learning Implementation",
            "AI Model Monitoring and Maintenance",
            "Prompt Engineering for LLMs",
            "Vector Databases and Embeddings",
            "AI Agent Development",
            "Multi-Modal AI Systems",
            # WEB DEVELOPMENT
            "Modern Web Development Frameworks",
            "React Best Practices and Patterns",
            "Next.js for Full-Stack Development",
            "Vue.js Component Architecture",
            "Angular Enterprise Applications",
            "Svelte and Modern JavaScript",
            "TypeScript for Large-Scale Apps",
            "WebAssembly Performance Optimization",
            "Progressive Web Apps (PWA)",
            "Server-Side Rendering (SSR)",
            "Static Site Generation (SSG)",
            "Jamstack Architecture",
            "Responsive Web Design Techniques",
            "Web Accessibility Standards (WCAG)",
            "Web Performance Optimization",
            "CSS Grid and Flexbox Mastery",
            "Tailwind CSS Utility-First Design",
            "Web Components and Custom Elements",
            "GraphQL API Development",
            "Real-Time Web Applications",
            # BACKEND & SYSTEM DESIGN
            "Backend Architecture Patterns",
            "RESTful API Design Principles",
            "Microservices vs Monolithic Architecture",
            "Event-Driven Architecture",
            "CQRS and Event Sourcing",
            "Database Design and Normalization",
            "SQL vs NoSQL Databases",
            "Database Indexing Strategies",
            "Caching Strategies (Redis, Memcached)",
            "Message Queues and Async Processing",
            "Load Balancing Techniques",
            "API Gateway Patterns",
            "Rate Limiting and Throttling",
            "Distributed Systems Design",
            "Scalability Patterns",
            "High Availability Systems",
            "Disaster Recovery Planning",
            "Database Replication and Sharding",
            "Serverless Architecture Patterns",
            "Backend Testing Strategies",
            # DEVOPS & CLOUD
            "DevOps Best Practices and Culture",
            "CI/CD Pipeline Implementation",
            "Docker Containerization Guide",
            "Kubernetes Orchestration",
            "Infrastructure as Code (Terraform)",
            "AWS Cloud Architecture",
            "Azure Cloud Services",
            "Google Cloud Platform (GCP)",
            "Cloud Cost Optimization",
            "GitOps Workflow Implementation",
            "Monitoring and Observability",
            "Log Management Solutions",
            "Cloud Security Best Practices",
            "Service Mesh Architecture",
            "Secrets Management",
            "Blue-Green Deployment",
            "Canary Deployments",
            "Cloud Migration Strategies",
            "Multi-Cloud Architecture",
            "Site Reliability Engineering (SRE)",
            # CYBERSECURITY
            "Cybersecurity Fundamentals",
            "Zero Trust Security Architecture",
            "Application Security Testing (SAST/DAST)",
            "API Security Best Practices",
            "OAuth 2.0 and OpenID Connect",
            "JWT Authentication Implementation",
            "Encryption and Key Management",
            "Web Application Firewall (WAF)",
            "DDoS Protection Strategies",
            "Penetration Testing Methodologies",
            "Security Compliance (SOC 2, ISO 27001)",
            "Vulnerability Management",
            "Incident Response Planning",
            "Security Monitoring and SIEM",
            "Container Security Best Practices",
            # DATA ENGINEERING & ANALYTICS
            "Data Engineering Pipelines",
            "ETL vs ELT Processes",
            "Data Warehousing Solutions",
            "Data Lake Architecture",
            "Real-Time Data Processing",
            "Apache Kafka for Streaming",
            "Apache Spark Big Data Processing",
            "Data Quality Management",
            "Data Governance Frameworks",
            "Business Intelligence Tools",
            "Data Visualization Best Practices",
            "A/B Testing and Experimentation",
            "Data Mesh Architecture",
            "Delta Lake and Data Lakehouse",
            "Snowflake Cloud Data Platform",
            # MOBILE & CROSS-PLATFORM
            "Native iOS Development with Swift",
            "Android Development with Kotlin",
            "React Native Cross-Platform Apps",
            "Flutter Mobile Development",
            "Mobile App Architecture Patterns",
            "Mobile UI/UX Best Practices",
            "Mobile Performance Optimization",
            "Push Notifications Implementation",
            "Mobile App Security",
            "App Store Optimization (ASO)",
            "Mobile Backend as a Service (BaaS)",
            "Mobile CI/CD Automation",
            # EMERGING TECHNOLOGIES
            "Blockchain Technology Explained",
            "Smart Contract Development",
            "Web3 and Decentralized Apps (DApps)",
            "NFT Technology and Use Cases",
            "Cryptocurrency and Blockchain",
            "Internet of Things (IoT) Architecture",
            "IoT Device Management",
            "Edge Computing Applications",
            "5G Technology Impact",
            "Quantum Computing Basics",
            "Augmented Reality (AR) Development",
            "Virtual Reality (VR) Applications",
            "Digital Twin Technology",
            "Robotics Process Automation (RPA)",
            "Low-Code/No-Code Platforms",
            # SOFTWARE ENGINEERING PRACTICES
            "Clean Code Principles",
            "SOLID Design Principles",
            "Design Patterns in Practice",
            "Test-Driven Development (TDD)",
            "Behavior-Driven Development (BDD)",
            "Code Review Best Practices",
            "Refactoring Legacy Code",
            "Technical Debt Management",
            "Agile Development Methodologies",
            "Scrum vs Kanban",
            "Pair Programming Techniques",
            "Documentation Best Practices",
            # PERFORMANCE & OPTIMIZATION
            "Application Performance Monitoring",
            "Database Query Optimization",
            "Frontend Performance Tuning",
            "Image Optimization Techniques",
            "Lazy Loading Implementation",
            "Code Splitting Strategies",
            "Memory Management Best Practices",
            "Network Performance Optimization",
            "Algorithm Complexity Analysis",
            "Profiling and Benchmarking",
            # PROGRAMMING LANGUAGES
            "Python for Data Science",
            "JavaScript ES6+ Features",
            "TypeScript Advanced Types",
            "Go Programming Concurrency",
            "Rust Memory Safety",
            "Java Spring Boot Development",
            "C# .NET Core Applications",
            "PHP Modern Development",
            "Ruby on Rails Web Apps",
            "Kotlin for Backend Development",
            "Swift for iOS Development",
            "Functional Programming Concepts",
            # CAREER & SOFT SKILLS
            "Tech Interview Preparation Guide",
            "System Design Interview Tips",
            "Building a Tech Portfolio",
            "Remote Work Best Practices",
            "Tech Leadership Skills",
            "Open Source Contribution Guide",
            "Personal Branding for Developers",
            "Salary Negotiation for Tech Roles",
            # TOOLS & PRODUCTIVITY
            "VS Code Extensions for Productivity",
            "Git Advanced Techniques",
            "Command Line Productivity Tips",
            "Developer Workflow Automation",
            "Debugging Techniques",
            "API Testing Tools (Postman, Insomnia)",
            "Database Management Tools",
            "Project Management for Developers",
            "Time Management for Engineers",
            "Technical Writing Skills",
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
            print("\nAPI chain: Groq (primary) -> OpenRouter (fallback) -> local template")
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
                topic     = pick_next_topic()
                blog_post = asyncio.run(blog_system.generate_blog_post(topic))
                blog_system.save_post(blog_post)

                generator = StaticSiteGenerator(blog_system)
                generator.generate_site()

                print(f"Post '{blog_post.title}' generated successfully!")

                visibility = VisibilityAutomator(config)

                hashtags   = blog_post.twitter_hashtags if hasattr(blog_post, 'twitter_hashtags') else ""
                tweet_text = (
                    f"New Post: {blog_post.title}\n\n"
                    f"{blog_post.meta_description[:100]}...\n\n"
                    f"Read more: https://kubaik.github.io/{blog_post.slug}\n\n"
                    f"{hashtags}"
                )
                if len(tweet_text) > 280:
                    tweet_text = (
                        f"{blog_post.title}\n"
                        f"Read: https://kubaik.github.io/{blog_post.slug}\n"
                        f"{hashtags}"
                    )

                twitter_result = visibility.post_with_best_strategy(blog_post)
                if twitter_result['success']:
                    print(f"Tweeted successfully: {twitter_result['url']}")
                else:
                    print(f"Twitter post failed: {twitter_result.get('error')}")

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
            generator   = StaticSiteGenerator(blog_system)
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
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        post_json   = item / "post.json"
                        post_md     = item / "index.md"
                        social_json = item / "social_posts.json"
                        print(f"    post.json:         {'Yes' if post_json.exists()   else 'No'}")
                        print(f"    index.md:          {'Yes' if post_md.exists()     else 'No'}")
                        print(f"    social_posts.json: {'Yes' if social_json.exists() else 'No'}")
                        if post_json.exists():
                            try:
                                with open(post_json, 'r') as f:
                                    data = json.load(f)
                                print(f"    Valid post:        {data.get('title', 'Unknown')}")
                                print(f"    Affiliate links:   {len(data.get('affiliate_links', []))}")
                                print(f"    Ad slots:          {data.get('monetization_data', {}).get('ad_slots', 0)}")
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
            generator   = StaticSiteGenerator(blog_system)
            posts       = generator._get_all_posts()

            visibility = VisibilityAutomator(config)

            print(f"Generating social media posts for {len(posts)} posts...")
            for post in posts:
                social_posts = visibility.generate_social_posts(post)

                post_dir    = blog_system.output_dir / post.slug
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

            visibility      = VisibilityAutomator(config)
            connection_test = visibility.test_twitter_connection()
            print(f"Connection test: {connection_test}")

            if connection_test['success']:
                class TestPost:
                    def __init__(self):
                        self.title            = "Test - AI Blog System Twitter Integration"
                        self.meta_description = "Testing our automated blog to Twitter posting system."
                        self.slug             = "test-twitter-integration"
                        self.tags             = ["test", "automation", "blogging"]

                test_post    = TestPost()
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
            args = ["python", "deduplicate_posts.py", "--delete"] + sys.argv[2:]
            subprocess.run(args)

        else:
            print("Usage: python blog_system.py [init|auto|build|cleanup|debug|social|test-twitter|dedup]")
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