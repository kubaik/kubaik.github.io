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

_STOP_WORDS = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is",
    "are", "with", "how", "your", "my", "our", "its", "on", "at",
    "by", "from", "this", "that", "best", "using", "guide", "complete",
    "introduction", "overview", "tutorial", "tips", "top", "ways",
}

DUPLICATE_TITLE_THRESHOLD = 0.35

MIN_WORD_COUNT = 1500


def _tokenise(text: str) -> set:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_duplicate_title(new_title: str, existing_titles: List[str],
                        threshold: float = DUPLICATE_TITLE_THRESHOLD) -> tuple:
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


def _count_words(text: str) -> int:
    return len(text.split())


# ─────────────────────────────────────────────────────────────────
# BlogSystem
# ─────────────────────────────────────────────────────────────────

class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)

        # ── API keys ──────────────────────────────────────────────
        self.groq_key = os.getenv("GROQ_API_KEY")
        # HuggingFace → SambaNova
        self.hf_token = os.getenv("HF_TOKEN")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        self._log_key_status()

        # Primary key for legacy compatibility checks
        self.api_key = (
            self.groq_key
            or self.hf_token
            or self.openrouter_key
            or self.gemini_key
        )

        self.monetization = MonetizationManager(config)
        self.hashtag_manager = HashtagManager(config)

    def _log_key_status(self):
        print("=== API Key Status ===")
        print(
            f"  Groq:             {'configured' if self.groq_key       else 'NOT SET'}")
        print(
            f"  HuggingFace (HF): {'configured' if self.hf_token       else 'NOT SET'}")
        print(
            f"  OpenRouter:       {'configured' if self.openrouter_key  else 'NOT SET'}")
        print(
            f"  Gemini:           {'configured' if self.gemini_key      else 'NOT SET'}")
        print("======================")

    # ─────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────

    def cleanup_posts(self):
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
    # API FALLBACK CHAIN:
    #   Groq → HuggingFace/Llama → OpenRouter → Gemini → local template
    # ─────────────────────────────────────────────────────────────

    async def _call_api_with_fallback(self, messages: List[Dict], max_tokens: int = 4000) -> str:
        providers = []

        if self.groq_key:
            providers.append(("Groq",              self._call_groq))
        if self.hf_token:
            providers.append(("HuggingFace/Llama", self._call_huggingface))
        if self.openrouter_key:
            providers.append(("OpenRouter",         self._call_openrouter))
        if self.gemini_key:
            providers.append(("Gemini",             self._call_gemini))

        if not providers:
            raise Exception(
                "No API keys configured. "
                "Set at least one of: GROQ_API_KEY, HF_TOKEN, "
                "OPENROUTER_API_KEY, GEMINI_API_KEY."
            )

        last_error = None
        for name, caller in providers:
            try:
                result = await caller(messages, max_tokens)
                print(f"API: {name} responded successfully.")
                return result
            except Exception as e:
                last_error = e
                print(f"{name} error: {e}")
                if name != providers[-1][0]:
                    print(f"Falling back to next provider...")

        raise Exception(
            f"All configured API providers failed. Last error: {last_error}\n"
            "Ensure at least one of GROQ_API_KEY / HF_TOKEN / "
            "OPENROUTER_API_KEY / GEMINI_API_KEY is set as a GitHub secret."
        )

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: Groq
    # ─────────────────────────────────────────────────────────────

    async def _call_groq(self, messages: List[Dict], max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    raise Exception(f"Groq {response.status}: {await response.text()}")
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: HuggingFace → SambaNova (Llama 3.3 70B Instruct)
    # Uses huggingface_hub InferenceClient when installed; falls back
    # to the OpenAI-compatible HF router REST API automatically.
    # pip install --upgrade huggingface_hub
    # GitHub secret: HF_TOKEN
    # ─────────────────────────────────────────────────────────────

    async def _call_huggingface(self, messages: List[Dict], max_tokens: int) -> str:
        HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
        HF_PROVIDER = "sambanova"
        HF_BASE_URL = "https://router.huggingface.co/v1"

        # ── Try huggingface_hub InferenceClient first ─────────────
        try:
            from huggingface_hub import InferenceClient

            def _sdk_call():
                client = InferenceClient(
                    provider=HF_PROVIDER,
                    token=self.hf_token,
                )
                response = client.chat.completions.create(
                    model=HF_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return response.choices[0].message.content

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _sdk_call)
            return result

        except ImportError:
            # huggingface_hub not installed — fall through to REST router
            pass

        # ── HuggingFace OpenAI-compatible REST router ─────────────
        # Router requires "model-id:provider" format in the model field
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type":  "application/json",
        }
        data = {
            "model":       f"{HF_MODEL}:{HF_PROVIDER}",
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": 0.7,
        }
        max_attempts = 3
        wait_seconds = [3, 8]

        for attempt in range(1, max_attempts + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{HF_BASE_URL}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=90),
                    ) as response:
                        if response.status != 200:
                            raise Exception(
                                f"HuggingFace {response.status}: {await response.text()}"
                            )
                        result = await response.json()
                        if "error" in result:
                            raise Exception(
                                f"HuggingFace error payload: {result['error']}"
                            )
                        return result["choices"][0]["message"]["content"]

            except aiohttp.ClientConnectionError as e:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"HuggingFace connection error "
                        f"(attempt {attempt}/{max_attempts}): {e}"
                    )
                    print(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"HuggingFace connection failed after {max_attempts} attempts: {e}"
                    )

            except asyncio.TimeoutError:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"HuggingFace timeout (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"HuggingFace timed out after {max_attempts} attempts."
                    )

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: OpenRouter
    # ─────────────────────────────────────────────────────────────

    async def _call_openrouter(self, messages: List[Dict], max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  self.config.get("base_url", "https://kubaik.github.io"),
            "X-Title":       self.config.get("site_name", "Tech Blog"),
        }
        data = {
            "model":       "openai/gpt-4o-mini",
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": 0.7,
            "provider": {
                "ignore":          ["Venice"],
                "allow_fallbacks": True,
            },
        }

        max_attempts = 3
        wait_seconds = [3, 8]

        for attempt in range(1, max_attempts + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        if response.status != 200:
                            raise Exception(
                                f"OpenRouter {response.status}: {await response.text()}"
                            )
                        result = await response.json()
                        if "error" in result:
                            raise Exception(
                                f"OpenRouter error payload: {result['error']}"
                            )
                        return result["choices"][0]["message"]["content"]

            except aiohttp.ClientConnectionError as e:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"OpenRouter connection error (attempt {attempt}/{max_attempts}): {e}")
                    print(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"OpenRouter connection failed after {max_attempts} attempts: {e}"
                    )

            except asyncio.TimeoutError:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"OpenRouter timeout (attempt {attempt}/{max_attempts}). Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"OpenRouter timed out after {max_attempts} attempts."
                    )

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: Gemini (Google Generative Language REST API)
    # Uses the google-generativeai SDK when available; falls back to
    # plain REST so the system works without the package installed.
    # ─────────────────────────────────────────────────────────────

    async def _call_gemini(self, messages: List[Dict], max_tokens: int) -> str:
        GEMINI_MODEL = "gemini-2.5-flash"  # single source of truth — update here only

        # ── Try google-generativeai SDK first ────────────────────
        try:
            import google.generativeai as genai

            def _sdk_call():
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,  # was hardcoded "gemini-1.5-flash"
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7,
                    ),
                )

                parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prefix = "SYSTEM: " if role == "system" else "USER: "
                    parts.append(f"{prefix}{content}")

                prompt = "\n\n".join(parts) + "\n\nASSISTANT:"
                response = model.generate_content(prompt)
                return response.text

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _sdk_call)
            return result

        except ImportError:
            pass

        # ── Gemini REST API — v1 (not v1beta) ────────────────────
        # v1beta dropped support for 1.5 models; v1 is the stable endpoint
        api_url = (
            f"https://generativelanguage.googleapis.com/v1/models/"  # was v1beta
            f"{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
        )

        system_parts = [
            msg["content"] for msg in messages if msg.get("role") == "system"
        ]
        user_parts = [
            msg["content"] for msg in messages if msg.get("role") != "system"
        ]

        first_user = ""
        if system_parts:
            first_user = "\n\n".join(system_parts) + "\n\n"
        first_user += (user_parts[0] if user_parts else "")

        contents = [{"role": "user", "parts": [{"text": first_user}]}]
        for extra in user_parts[1:]:
            contents.append({"role": "user", "parts": [{"text": extra}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature":     0.7,
            },
        }

        max_attempts = 3
        wait_seconds = [3, 8]

        for attempt in range(1, max_attempts + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=90),
                    ) as response:
                        if response.status != 200:
                            raise Exception(
                                f"Gemini {response.status}: {await response.text()}"
                            )
                        result = await response.json()

                        try:
                            return (
                                result["candidates"][0]["content"]["parts"][0]["text"]
                            )
                        except (KeyError, IndexError) as parse_err:
                            raise Exception(
                                f"Gemini unexpected response shape: {parse_err} — {result}"
                            )

            except aiohttp.ClientConnectionError as e:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"Gemini connection error (attempt {attempt}/{max_attempts}): {e}")
                    print(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"Gemini connection failed after {max_attempts} attempts: {e}"
                    )

            except asyncio.TimeoutError:
                if attempt < max_attempts:
                    wait = wait_seconds[attempt - 1]
                    print(
                        f"Gemini timeout (attempt {attempt}/{max_attempts}). Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise Exception(
                        f"Gemini timed out after {max_attempts} attempts.")

    # ─────────────────────────────────────────────────────────────
    # CONTENT GENERATION
    # ─────────────────────────────────────────────────────────────

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            print("No API keys configured. Using local template content.")
            return self._generate_fallback_post(topic)

        try:
            print(f"Generating content for: {topic}")

            existing_titles = _load_existing_titles(self.output_dir)
            title = await self._generate_unique_title(topic, keywords, existing_titles)

            content = await self._generate_content(title, topic, keywords)
            meta_description = await self._generate_meta_description(topic, title)
            slug = self._create_slug(title)

            if not keywords:
                keywords = await self._generate_keywords(topic, title)

            word_count = _count_words(content)
            print(f"Generated content: {word_count} words")
            if word_count < MIN_WORD_COUNT:
                print(
                    f"Warning: content only {word_count} words "
                    f"(min {MIN_WORD_COUNT}). Expanding..."
                )
                content = await self._expand_content(content, title, topic)
                word_count = _count_words(content)
                print(f"After expansion: {word_count} words")

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
                monetization_data={},
            )

            enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
                post.content, topic
            )
            post.content = enhanced_content
            post.affiliate_links = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(
                enhanced_content)

            print("Generating trending hashtags...")
            hashtags = await self.hashtag_manager.get_daily_hashtags(
                topic, max_hashtags=10
            )
            post.tags = list(set(post.tags + hashtags))[:15]
            post.seo_keywords = list(set(post.seo_keywords + hashtags))[:15]
            post.twitter_hashtags = self.hashtag_manager.format_hashtags_for_twitter(
                hashtags[:5]
            )
            print(f"Hashtags: {', '.join(hashtags[:5])}")

            return post

        except Exception as e:
            print(
                f"All API providers exhausted ({e}). Using local template content.")
            return self._generate_fallback_post(topic)

    # ─────────────────────────────────────────────────────────────
    # INDIVIDUAL GENERATION STEPS
    # ─────────────────────────────────────────────────────────────

    async def _generate_unique_title(self, topic: str, keywords: List[str],
                                     existing_titles: List[str],
                                     max_attempts: int = 3) -> str:
        for attempt in range(1, max_attempts + 1):
            extra_instruction = ""
            if attempt > 1:
                extra_instruction = (
                    " IMPORTANT: Do NOT produce a title similar to any of these: "
                    + ", ".join(f'"{t}"' for t in existing_titles[:10])
                    + ". Choose a clearly different angle."
                )

            title = await self._generate_title(
                topic, keywords, extra_instruction=extra_instruction
            )
            title = title.strip().strip('"')

            if not existing_titles:
                return title

            is_dup, match, score = _is_duplicate_title(title, existing_titles)
            if not is_dup:
                print(
                    f"Title accepted (similarity {score:.0%} to nearest existing): {title}"
                )
                return title

            print(
                f"Attempt {attempt}: generated title is too similar "
                f"({score:.0%}) to existing post '{match}'. Retrying…"
            )

        print("Warning: could not generate a fully unique title. Appending date suffix.")
        suffix = f" ({datetime.now().strftime('%B %Y')})"
        return f"{title}{suffix}"

    async def _generate_title(self, topic: str, keywords: List[str] = None,
                              extra_instruction: str = "") -> str:
        keyword_text = f" Focus on keywords: {', '.join(keywords)}" if keywords else ""
        messages = [
            {
                "role": "system",
                "content": "You are a skilled blog title writer. Create engaging, SEO-friendly titles.",
            },
            {
                "role": "user",
                "content": (
                    f"Generate a compelling blog post title about '{topic}'.{keyword_text} "
                    "The title should be catchy, informative, and under 60 characters. "
                    "Avoid generic titles starting with 'The Ultimate Guide' or "
                    "'Everything You Need to Know'."
                    f"{extra_instruction}"
                ),
            },
        ]
        title = await self._call_api_with_fallback(messages, max_tokens=100)
        return title.strip().strip('"')

    async def _generate_content(self, title: str, topic: str,
                                keywords: List[str] = None) -> str:
        keyword_text = (
            f"\nKeywords to incorporate naturally: {', '.join(keywords)}"
            if keywords else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced tech professional with 10+ years of hands-on experience. "
                    "Write in a direct, opinionated voice — share specific insights, real tradeoffs, "
                    "and concrete numbers. "
                    "Never use filler phrases like 'in today's fast-paced world', 'crucial aspect', "
                    "or 'it is important to note'. "
                    "Every paragraph must deliver concrete value. Be specific: name actual tools, "
                    "libraries, companies, and version numbers. "
                    "Take clear stances. Acknowledge tradeoffs honestly."
                ),
            },
            {
                "role": "user",
                "content": f"""Write a 2000-word technical blog post with the title: \"{title}\"

Topic: {topic}{keyword_text}

Structure (use exactly these ## headings):
## The Problem Most Developers Miss
## How [Topic] Actually Works Under the Hood
## Step-by-Step Implementation
## Real-World Performance Numbers
## Common Mistakes and How to Avoid Them
## Tools and Libraries Worth Using
## When Not to Use This Approach
## Conclusion and Next Steps

Requirements:
- Write in Markdown format
- Include at least 1 realistic code example (with language tag, e.g. ```python)
- Include specific tool names with version numbers where relevant
- Include at least 2 concrete numbers (benchmarks, percentages, file sizes, etc.)
- Each section must be at minimum 150 words
- Address a real pain point developers encounter
- The "When Not to Use This Approach" section must be honest and specific

Avoid:
- Vague phrases like "significantly improve", "seamlessly integrate", "powerful solution"
- Padding sentences that add length without adding information
- Lists of more than 6 items (they read like AI output)
- Starting sentences with "In conclusion" or "Overall"
- The phrase "dive into" or "delve into"

Do not include the main title (# {title}) — it is added automatically.""",
            },
        ]
        content = await self._call_api_with_fallback(messages, max_tokens=4000)
        return content.strip()

    async def _expand_content(self, existing_content: str, title: str, topic: str) -> str:
        """Expand thin content by adding additional substantive sections."""
        messages = [
            {
                "role": "system",
                "content": "You are a technical writer expanding existing blog content with substantive additions.",
            },
            {
                "role": "user",
                "content": (
                    f"The following blog post about '{topic}' is too short. "
                    "Add 3 additional detailed sections at the end (each 200+ words) covering:\n"
                    "1. Advanced configuration and edge cases\n"
                    "2. Integration with popular existing tools or workflows\n"
                    "3. A realistic case study or before/after comparison\n\n"
                    f"Existing content:\n{existing_content}\n\n"
                    "Return the complete article including original content plus the new sections. "
                    "Do not include the title line."
                ),
            },
        ]
        return await self._call_api_with_fallback(messages, max_tokens=4000)

    async def _generate_meta_description(self, topic: str, title: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You create SEO-optimized meta descriptions that are specific and enticing.",
            },
            {
                "role": "user",
                "content": (
                    f"Write a meta description (under 155 characters) for a blog post "
                    f"titled '{title}' about {topic}. "
                    "Be specific — mention what the reader will learn or gain. "
                    "Do not start with 'Learn how to' or 'Discover'. Avoid generic phrases."
                ),
            },
        ]
        description = await self._call_api_with_fallback(messages, max_tokens=100)
        return description.strip().strip('"')

    async def _generate_keywords(self, topic: str, title: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "You generate relevant SEO keywords for technical blog posts.",
            },
            {
                "role": "user",
                "content": (
                    f"Generate 8-10 relevant SEO keywords for a blog post titled '{title}' "
                    f"about {topic}. "
                    "Include a mix of: 2 short-tail keywords, 4 long-tail keywords, "
                    "2 question-based keywords. "
                    "Return as a comma-separated list, no numbering."
                ),
            },
        ]
        keywords_text = await self._call_api_with_fallback(messages, max_tokens=150)
        keywords = [k.strip().strip('"') for k in keywords_text.split(',')]
        return [k for k in keywords if k][:10]

    # ─────────────────────────────────────────────────────────────
    # LOCAL FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _generate_fallback_post(self, topic: str) -> BlogPost:
        """Generate a structured fallback post when all APIs are unavailable."""
        title = f"{topic}: A Practical Technical Guide"
        slug = self._create_slug(title)
        topic_lower = topic.lower()
        topic_slug = topic.replace(' ', '').replace('-', '')[:20]

        content = f"""## The Problem Most Developers Miss

When working with {topic}, most developers jump straight to implementation without understanding the underlying mechanics. This leads to brittle solutions that fail under load, are difficult to debug, and create maintenance headaches down the line.

The most common mistake is treating {topic} as a black box. You configure it, it works in development, and you ship it — until production load reveals gaps in your assumptions. This guide covers what the documentation usually skips.

Before writing a single line of code, you need to answer three questions: What failure modes does {topic} introduce? What are the actual resource costs at scale? And what does the fallback look like when it fails?

## How {topic} Actually Works Under the Hood

At its core, {topic} relies on a combination of in-memory state management and persistent coordination. Understanding this dual nature is the key to avoiding the most common performance problems.

When a request comes in, the system first checks local state (fast, ~1ms), then falls back to shared state (slower, typically 10–50ms depending on network conditions). Most documentation focuses on the happy path. Real systems need to handle the cases where shared state is unavailable, inconsistent, or outdated.

The coordination overhead is real. In benchmarks across several production systems, poorly configured {topic} setups added 15–40% latency compared to a baseline. Well-tuned implementations added 2–8%. The difference is almost entirely in how you handle connection pooling and retry logic.

Memory usage scales roughly linearly with concurrent connections. Budget approximately 2–5MB per 100 active connections for the coordination layer. This is separate from your application memory and often overlooked in capacity planning.

## Step-by-Step Implementation

Here is a minimal, production-ready implementation pattern:

```python
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class {topic_slug}Client:
    def __init__(self, config: Dict[str, Any]):
        self.config      = config
        self.max_retries = config.get("max_retries", 3)
        self.timeout     = config.get("timeout_seconds", 5.0)
        self._connection = None

    def connect(self) -> bool:
        \"\"\"Establish connection with exponential backoff.\"\"\"
        for attempt in range(self.max_retries):
            try:
                self._connection = self._create_connection()
                logger.info(f"Connected on attempt {{attempt + 1}}")
                return True
            except ConnectionError as e:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Attempt {{attempt + 1}} failed: {{e}}. Retrying in {{wait}}s")
                time.sleep(wait)
        return False

    def _create_connection(self):
        raise NotImplementedError

    def health_check(self) -> bool:
        if self._connection is None:
            return False
        try:
            return self._ping()
        except Exception:
            self._connection = None
            return False
```

Step 1: Install dependencies and set environment variables. Never hardcode credentials — use environment variables or a secrets manager.

Step 2: Initialise with conservative timeouts. Start at 5 seconds and tune down based on your p99 latency measurements.

Step 3: Add circuit breaker logic around all external calls. After 5 consecutive failures, stop trying for 30 seconds.

Step 4: Instrument everything. Track: connection attempt count, success rate, p50/p95/p99 latency, and error rates by error type.

Step 5: Load test before going live with realistic traffic patterns, not just peak load.

## Real-World Performance Numbers

Based on production deployments across different scales:

- **Small scale (under 1,000 req/min):** Overhead is negligible. Default configuration works fine. Focus on correctness, not optimisation.
- **Medium scale (1,000–50,000 req/min):** Connection pooling becomes critical. Without it, expect 20–35% latency increase under load. Pool size: start at 10 connections per application instance.
- **Large scale (50,000+ req/min):** Single coordinator nodes become bottlenecks. Benchmarks show 40% throughput improvement moving from single-node to clustered setup.

Cold start latency is often 10× worse than steady-state. If your application auto-scales, build in a 2–3 second warmup period before routing traffic to new instances.

## Common Mistakes and How to Avoid Them

**Mistake 1: No timeout on individual operations.** Most libraries default to no timeout or 30+ seconds. Set explicit timeouts: connection timeout (2–5s) and per-operation timeout (1–5s).

**Mistake 2: Treating errors as binary.** A connection refused error warrants a different response than a timeout, which differs from an authentication error. Build specific handlers for each error class.

**Mistake 3: No connection pool monitoring.** Pool exhaustion causes requests to queue silently. Add metrics for pool size, active connections, waiting requests, and wait time. Alert when wait time exceeds 500ms.

**Mistake 4: Testing only the happy path.** Use fault injection in staging: simulate network partitions, slow responses, and connection drops. Most production incidents come from failure modes that were never tested.

**Mistake 5: Ignoring DNS caching.** In containerised environments, DNS records change frequently. Set TTL to 30–60 seconds, not 300+.

## Tools and Libraries Worth Using

- **Prometheus + Grafana:** Standard stack for metrics. Use histograms (not averages) for latency.
- **OpenTelemetry:** Distributed tracing. Adds ~1–2% overhead but invaluable for debugging.
- **Testcontainers:** Spin up real infrastructure in tests. Far better than mocks.
- **k6 or Locust:** Load testing. Run weekly against staging, not just before launch.
- **resilience4j (JVM) / tenacity (Python) / polly (.NET):** Ready-made circuit breaker and retry implementations.

## When Not to Use This Approach

This pattern is not the right choice in every situation:

**Skip it if your traffic is low and predictable.** Under 100 requests/minute with no spikes, the added complexity is not worth it.

**Skip it if you do not have observability in place.** Distributed systems require distributed tracing to debug. If you cannot see what is happening across service boundaries, you will spend more time debugging than you saved.

**Skip it if your team is unfamiliar with the failure modes.** Operational complexity is a real cost. A simpler system your team understands deeply will outperform a sophisticated one that confuses them.

**Consider alternatives when:** strong consistency is required, latency budget is extremely tight (sub-millisecond), or you are operating in environments with unreliable networking.

## Conclusion and Next Steps

The gap between a working prototype and a production-ready {topic} implementation comes down to handling failure cases systematically. The happy path is easy. The value is in what happens when things go wrong.

Three actions to take now: add explicit timeouts to every operation today; set up latency histograms (p50, p95, p99) this week; run a chaos test against staging this month.

Further reading: the official {topic} documentation covers configuration options in depth. For production patterns, the Google SRE book chapters on managing risk and cascading failures are directly applicable."""

        post = BlogPost(
            title=title,
            content=content,
            slug=slug,
            tags=[
                topic_lower.replace(' ', '-'),
                'development',
                'technical-guide',
                'best-practices',
            ],
            meta_description=(
                f"A practical guide to {topic} covering implementation, "
                "real performance benchmarks, common mistakes, and honest tradeoffs."
            ),
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            seo_keywords=[
                topic_lower,
                f"{topic_lower} tutorial",
                f"{topic_lower} best practices",
                f"how to use {topic_lower}",
                f"{topic_lower} performance",
            ],
            affiliate_links=[],
            monetization_data={},
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
        word_count = _count_words(post.content)
        if word_count < MIN_WORD_COUNT:
            print(
                f"Warning: saving post with only {word_count} words "
                f"(min recommended: {MIN_WORD_COUNT})"
            )

        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post.to_dict(), f, indent=2, ensure_ascii=False)

        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")

        print(f"Saved post: {post.title} ({post.slug}) — {word_count} words")
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
            f"Config file {config_path} not found. "
            "Run 'python blog_system.py init' first."
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

    docs_dir = Path("./docs")
    existing_titles = _load_existing_titles(docs_dir)

    if existing_titles:
        safe_available = []
        skipped = []
        for candidate in available:
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
            print("All available topics are already covered. Resetting topic history.")
            available = topics
            used = []

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
            "facebook": "your-facebook-page",
        },

        "content_topics": [

            # ══════════════════════════════════════════════════════════════════
            # TIER 1 — HIGHEST VIRALITY (broad audience, developers + non-devs)
            # ══════════════════════════════════════════════════════════════════

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
            # TIER 2 — STRONG VIRALITY (developer-focused)
            # ══════════════════════════════════════════════════════════════════

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
            # TIER 3 — SOLID SEO
            # ══════════════════════════════════════════════════════════════════

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
        ],
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("Created sample config.yaml file with monetization settings")
    print("\nNext steps:")
    print("1. Replace 'your-tag-20' with your Amazon Associates tag")
    print("2. Add your Google Analytics 4 measurement ID")
    print("3. Add your Google AdSense ID (ca-pub-xxxxxxxxxx)")
    print("4. Update social media handles")
    print("5. Add GitHub secrets:")
    print("     GROQ_API_KEY       (primary)")
    print("     HF_TOKEN           (fallback 1 — HuggingFace/SambaNova Llama 3.3 70B)")
    print("     OPENROUTER_API_KEY (fallback 2)")
    print("     GEMINI_API_KEY     (fallback 3)")


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
                "\nAPI chain: Groq → HuggingFace/Llama → OpenRouter → Gemini → local template"
            )
            print(
                "Add GitHub secrets: GROQ_API_KEY, HF_TOKEN, "
                "OPENROUTER_API_KEY, GEMINI_API_KEY"
            )

        elif mode == "auto":
            print("Starting automated blog generation...")
            print(
                "API chain: Groq → HuggingFace/Llama → OpenRouter → Gemini → local template")

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

                print("\nPosting as thread for maximum impressions...")
                thread_result = visibility.post_thread(blog_post)

                if thread_result['success']:
                    print(
                        f"Thread posted ({thread_result['tweet_count']} tweets)")
                    print(f"   First tweet: {thread_result['first_tweet']}")
                    for i, url in enumerate(thread_result['thread_urls'], 1):
                        print(f"   Tweet {i}: {url}")

                    try:
                        import time
                        import datetime as dt_module

                        time.sleep(3)

                        today = dt_module.date.today().isoformat()
                        flag_file = f".last_reply_{today}"

                        if not os.path.exists(flag_file):
                            last_tweet_id = thread_result['thread_ids'][-1]
                            username = config.get('social_accounts', {}).get(
                                'twitter', '@KubaiKevin'
                            )

                            hashtags = ""
                            if hasattr(blog_post, 'tags') and blog_post.tags:
                                sorted_tags = sorted(
                                    blog_post.tags, key=len)[:3]
                                hashtags = " ".join(
                                    f"#{t.replace(' ', '').replace('-', '')}"
                                    for t in sorted_tags if t
                                )

                            followup_text = (
                                f"Found this useful? Follow {username} for daily threads on "
                                f"AI, dev tools, and software engineering\n\n"
                                f"{hashtags}"
                            ).strip()

                            if len(followup_text) > 280:
                                followup_text = followup_text[:277] + "..."

                            followup_response = visibility.twitter_client.create_tweet(
                                text=followup_text,
                                in_reply_to_tweet_id=last_tweet_id,
                            )
                            followup_id = followup_response.data['id']
                            twitter_user = visibility._username or "KubaiKevin"
                            followup_url = (
                                f"https://twitter.com/{twitter_user}/status/{followup_id}"
                            )

                            open(flag_file, 'w').close()
                            print(
                                f"Follow-up reply added to thread: {followup_url}")
                        else:
                            print("Follow-up reply already posted today, skipping.")

                    except Exception as followup_err:
                        print(
                            f"Follow-up reply failed (skipping): {followup_err}")

                else:
                    print(
                        f"Thread failed ({thread_result.get('error')}). "
                        "Falling back to single tweet..."
                    )
                    single_result = visibility.post_with_best_strategy(
                        blog_post)
                    if single_result['success']:
                        print(f"Single tweet posted: {single_result['url']}")
                    else:
                        print(
                            f"Single tweet also failed: {single_result.get('error')}")

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
                                wc = _count_words(data.get('content', ''))
                                print(
                                    f"    Valid post:        {data.get('title', 'Unknown')}")
                                print(
                                    f"    Word count:        {wc} "
                                    f"{'✓' if wc >= MIN_WORD_COUNT else '⚠ TOO SHORT'}"
                                )
                                print(
                                    f"    Affiliate links:   {len(data.get('affiliate_links', []))}")
                                print(
                                    f"    Ad slots:          "
                                    f"{data.get('monetization_data', {}).get('ad_slots', 0)}"
                                )
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
            print("Running deduplication...")
            import subprocess
            args = ["python", "deduplicate_posts.py",
                    "--delete"] + sys.argv[2:]
            subprocess.run(args)

        else:
            print(
                "Usage: python blog_system.py "
                "[init|auto|build|cleanup|debug|social|test-twitter|dedup]"
            )
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
        print(
            "API chain: Groq (primary) → HuggingFace/Llama → OpenRouter → Gemini → local template"
        )
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
        print("\nGitHub secrets (set at least one):")
        print("  GROQ_API_KEY       - Primary  (100k tokens/day free, very fast)")
        print("  HF_TOKEN           - Fallback 1 (HuggingFace/SambaNova — Llama 3.3 70B)")
        print("  OPENROUTER_API_KEY - Fallback 2 (GPT-4o-mini via OpenRouter)")
        print("  GEMINI_API_KEY     - Fallback 3 (Gemini 1.5 Flash — generous free tier)")
