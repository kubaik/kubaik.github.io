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

# ── CHANGE: raised from 1500 → 2000 to push toward genuinely long content ──
MIN_WORD_COUNT = 2000


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


# ── CHANGE: audit helper — find posts that used the fallback template ──
def audit_posts(docs_dir: Path) -> Dict:
    """
    Scan all saved posts and return counts + slugs of:
      - fallback posts  (contain the telltale 'used_fallback' flag or
        the literal placeholder class name produced by _generate_fallback_post)
      - short posts     (below MIN_WORD_COUNT)
    """
    results = {"fallback": [], "short": [], "ok": []}
    if not docs_dir.exists():
        return results
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name == "static":
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = data.get("content", "")
            wc = _count_words(content)
            is_fallback = (
                data.get("monetization_data", {}).get("used_fallback", False)
                or "class {topic_slug}Client" in content
                or "class Client:" in content and "max_retries = config.get" in content
            )
            if is_fallback:
                results["fallback"].append(post_dir.name)
            elif wc < MIN_WORD_COUNT:
                results["short"].append((post_dir.name, wc))
            else:
                results["ok"].append(post_dir.name)
        except Exception as e:
            print(f"Audit error for {post_dir.name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────
# Topic phrase extractor
# ─────────────────────────────────────────────────────────────────

_HOOK_STOP_WORDS = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is", "are",
    "with", "how", "your", "my", "our", "its", "on", "at", "by", "from",
    "this", "that", "best", "using", "guide", "complete", "introduction",
    "overview", "tutorial", "tips", "top", "ways", "actually", "really",
    "without", "beyond", "vs", "why", "when", "where", "which", "who",
    "most", "every", "what", "will", "does", "behind", "inside", "between",
    "about", "after", "before", "during", "through", "across",
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
    "burnout":          "Developer Burnout",
    "remote work":      "Remote Work",
    "tech salar":       "Tech Salaries",
    "negotiate":        "Salary Negotiation",
    "ai ethics":        "AI Ethics",
    "ai tool":          "AI Tools",
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
    import re as _re
    title_lower = f" {title.lower()} "
    for key, phrase in _TOPIC_OVERRIDES.items():
        if key in title_lower:
            return phrase
    cleaned = _re.sub(r"[^\w\s\-]", " ", title)
    words = cleaned.split()
    meaningful = []
    for w in words:
        if w.lower() in _HOOK_STOP_WORDS:
            continue
        if _re.match(r'^\d{4}$', w):
            continue
        if w.isupper() and len(w) >= 2:
            meaningful.append(w)
        elif len(w) >= 3:
            meaningful.append(w)
    if not meaningful:
        return title[:40]
    return " ".join(meaningful[:max_words])


# ─────────────────────────────────────────────────────────────────
# Tiered hashtag system  (unchanged)
# ─────────────────────────────────────────────────────────────────

_HASHTAG_TIERS = {
    "broad": {
        " ai ":           ["AI", "ArtificialIntelligence"],
        "artificial int": ["AI", "ArtificialIntelligence"],
        "python":         ["Python", "Python3"],
        "javascript":     ["JavaScript", "JS"],
        "typescript":     ["TypeScript"],
        "react":          ["ReactJS"],
        "frontend":       ["WebDev", "Frontend"],
        "backend":        ["Backend", "SoftwareEngineering"],
        " web ":          ["WebDev"],
        "web dev":        ["WebDev"],
        "devops":         ["DevOps"],
        "cloud":          ["CloudComputing"],
        "security":       ["CyberSecurity", "InfoSec"],
        "hacker":         ["CyberSecurity", "EthicalHacking"],
        "data ":          ["DataEngineering"],
        "data science":   ["DataScience"],
        "machine learn":  ["MachineLearning"],
        " ml ":           ["MachineLearning"],
        "llm":            ["LLM", "GenerativeAI"],
        "generat":        ["GenerativeAI"],
        " tech ":         ["Tech", "Technology"],
        "coding":         ["Coding", "Programming"],
        "programming":    ["Programming"],
        "software":       ["SoftwareEngineering"],
        "startup":        ["Startups", "Entrepreneurship"],
        " api ":          ["APIs"],
        "apis":           ["APIs"],
        "database":       ["Database"],
        "performance":    ["Performance"],
        "mobile":         ["MobileDev"],
        "android":        ["AndroidDev"],
        " ios ":          ["iOSDev"],
        "profit":         ["Entrepreneurship", "Tech"],
        "income":         ["PassiveIncome", "Entrepreneurship"],
        "salary":         ["TechCareer"],
        "career":         ["TechCareer"],
        "developer":      ["SoftwareEngineering", "Coding"],
        "engineer":       ["SoftwareEngineering"],
    },
    "niche": {
        "kubernetes":      ["Kubernetes", "K8s"],
        "docker":          ["Docker", "Containers"],
        "container":       ["Docker", "Containers"],
        "rustlang":        ["RustLang"],
        " rust ":          ["RustLang"],
        "golang":          ["Golang"],
        " go ":            ["Golang"],
        "java ":           ["Java"],
        "rest api":        ["REST", "APIDesign"],
        "graphql":         ["GraphQL"],
        " sql ":           ["SQL"],
        "postgres":        ["PostgreSQL"],
        "mysql":           ["MySQL"],
        "mongodb":         ["MongoDB"],
        "redis":           ["Redis"],
        "kafka":           ["ApacheKafka"],
        "system design":   ["SystemDesign"],
        "open source":     ["OpenSource"],
        "cloud native":    ["CloudNative"],
        "terraform":       ["Terraform", "IaC"],
        "github":          ["GitHub"],
        "swift":           ["Swift", "iOSDev"],
        "kotlin":          ["Kotlin", "AndroidDev"],
        "flutter":         ["Flutter"],
        "react native":    ["ReactNative"],
        "next.js":         ["NextJS"],
        "nextjs":          ["NextJS"],
        "tailwind":        ["TailwindCSS"],
        "serverless":      ["Serverless"],
        "microservice":    ["Microservices"],
        "rag":             ["RAG", "VectorSearch"],
        "vector":          ["VectorDB"],
        "saas":            ["SaaS"],
        "mlops":           ["MLOps"],
        "fine-tun":        ["FineTuning"],
        "gpt":             ["ChatGPT", "OpenAI"],
        "chatgpt":         ["ChatGPT"],
        "prompt engineer": ["PromptEngineering"],
        "penetration":     ["PenTesting"],
        "zero trust":      ["ZeroTrust"],
        "ci/cd":           ["CICD", "DevOps"],
        "gitops":          ["GitOps"],
        "websocket":       ["WebSockets", "RealTime"],
        "webassembly":     ["WebAssembly", "WASM"],
        "wasm":            ["WebAssembly"],
    },
    "monetization": {
        "passive income":    ["PassiveIncome"],
        "side hustle":       ["SideHustle"],
        "indie hacker":      ["IndieHacker"],
        " indie ":           ["IndieHacker"],
        "freelance":         ["Freelancing"],
        "build in public":   ["BuildInPublic"],
        "building in publi": ["BuildInPublic"],
        "bootstrapp":        ["BootstrappedFounder"],
        "product launch":    ["ProductLaunch"],
        " mvp":              ["BuildInPublic", "IndieHacker"],
        "monetize":          ["Monetization"],
        "affiliate":         ["AffiliateMarketing"],
        " blog":             ["Blogging", "ContentCreator"],
        "content creator":   ["ContentCreator"],
        "learn to code":     ["LearnToCode", "100DaysOfCode"],
        "get hired":         ["GetHired", "TechJobs"],
        " job":              ["TechJobs"],
        "remote work":       ["RemoteWork"],
        "digital nomad":     ["DigitalNomad"],
        "profit":            ["Entrepreneurship", "BuildInPublic"],
        "make money":        ["MakeMoneyOnline"],
        "10k":               ["IndieHacker", "MicroSaaS"],
        "150k":              ["TechSalary"],
        "negotiate":         ["CareerAdvice"],
        "promoted":          ["CareerAdvice", "TechCareer"],
        "burnout":           ["DevWellbeing"],
        "burn out":          ["DevWellbeing"],
    },
}


def _derive_hashtags_from_keywords(
    keywords: List[str],
    topic: str = "",
    title: str = "",
    max_hashtags: int = 5,
) -> List[str]:
    combined = f" {' '.join([title, topic] + keywords).lower()} "
    selected: Dict[str, List[str]] = {
        "broad": [], "niche": [], "monetization": []}
    for tier, mapping in _HASHTAG_TIERS.items():
        for keyword, tags in mapping.items():
            if keyword in combined:
                for tag in tags:
                    if tag not in selected[tier]:
                        selected[tier].append(tag)
    result: List[str] = []
    result.extend(selected["broad"][:2])
    result.extend(selected["niche"][:2])
    result.extend(selected["monetization"][:1])
    if len(result) < max_hashtags:
        question_starters = {"how", "what", "why",
                             "when", "where", "which", "who"}
        for kw in keywords:
            kw = kw.strip().lower()
            if not kw:
                continue
            words = kw.split()
            if words and words[0] in question_starters:
                continue
            if len(words) <= 2:
                tag = "".join(w.capitalize() for w in words)
                tag = re.sub(r"[^\w]", "", tag)
                if tag and tag not in result:
                    result.append(tag)
            if len(result) >= max_hashtags:
                break
    seen: set = set()
    final: List[str] = []
    for tag in result:
        if tag not in seen:
            seen.add(tag)
            final.append(tag)
    return final[:max_hashtags]


# ─────────────────────────────────────────────────────────────────
# Provider constants
# ─────────────────────────────────────────────────────────────────

_MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
_MISTRAL_MODEL = "mistral-large-latest"
_MISTRAL_FREE_TIER_DELAY = 1.2

_NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"


# ─────────────────────────────────────────────────────────────────
# BlogSystem
# ─────────────────────────────────────────────────────────────────

class BlogSystem:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)

        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.cerebras_key = os.getenv("CEREBRAS_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        self.nvidia_key = os.getenv("NVIDIA_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        self._log_key_status()

        self.api_key = (
            self.groq_key or self.openrouter_key or self.cerebras_key
            or self.mistral_key or self.nvidia_key or self.gemini_key
        )

        self.monetization = MonetizationManager(config)
        self.hashtag_manager = HashtagManager(config)

    def _log_key_status(self):
        print("=== API Key Status ===")
        print(
            f"  Groq:       {'configured' if self.groq_key       else 'NOT SET'}")
        print(
            f"  OpenRouter: {'configured' if self.openrouter_key  else 'NOT SET'}")
        print(
            f"  Cerebras:   {'configured' if self.cerebras_key    else 'NOT SET'}")
        print(
            f"  Mistral:    {'configured' if self.mistral_key     else 'NOT SET'}")
        print(
            f"  NVIDIA NIM: {'configured' if self.nvidia_key      else 'NOT SET'}")
        print(
            f"  Gemini:     {'configured' if self.gemini_key      else 'NOT SET'}")
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

    # ── CHANGE: new method to remove/flag low-quality posts before reapplying ──
    def purge_low_quality_posts(self, dry_run: bool = True):
        """
        Remove posts that are: (a) generated by the local fallback template,
        or (b) below MIN_WORD_COUNT.  Pass dry_run=False to actually delete.
        """
        results = audit_posts(self.output_dir)
        print(f"\n=== Post Quality Audit ===")
        print(f"  OK:       {len(results['ok'])} posts")
        print(f"  Short:    {len(results['short'])} posts")
        print(f"  Fallback: {len(results['fallback'])} posts")

        to_remove = results["fallback"] + \
            [slug for slug, _ in results["short"]]

        if not to_remove:
            print("Nothing to remove — all posts meet quality bar.")
            return

        for slug in to_remove:
            post_dir = self.output_dir / slug
            reason = "fallback" if slug in results["fallback"] else "too short"
            if dry_run:
                print(f"  [DRY RUN] Would remove: {slug} ({reason})")
            else:
                import shutil
                shutil.rmtree(post_dir, ignore_errors=True)
                print(f"  Removed: {slug} ({reason})")

        if dry_run:
            print(
                f"\nRun with dry_run=False to actually delete {len(to_remove)} posts.")
        else:
            print(f"\nPurged {len(to_remove)} low-quality posts.")

    # ─────────────────────────────────────────────────────────────
    # API FALLBACK CHAIN
    # ─────────────────────────────────────────────────────────────

    async def _call_api_with_fallback(self, messages: List[Dict], max_tokens: int = 4000) -> str:
        providers = []

        if self.groq_key:
            providers.append(("Groq",       self._call_groq))
        if self.mistral_key:
            providers.append(("Mistral",     self._call_mistral))
        if self.openrouter_key:
            providers.append(("OpenRouter",  self._call_openrouter))
        if self.cerebras_key:
            providers.append(("Cerebras",    self._call_cerebras))
        if self.gemini_key:
            providers.append(("Gemini",      self._call_gemini))
        if self.nvidia_key:
            providers.append(("NVIDIA NIM",  self._call_nvidia))

        if not providers:
            raise Exception(
                "No API keys configured. Set at least one of: GROQ_API_KEY, "
                "OPENROUTER_API_KEY, CEREBRAS_API_KEY, MISTRAL_API_KEY, "
                "NVIDIA_API_KEY, GEMINI_API_KEY."
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
                    print("Falling back to next provider...")

        raise Exception(
            f"All configured API providers failed. Last error: {last_error}")

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: Groq
    # ─────────────────────────────────────────────────────────────

    async def _call_groq(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.groq_key}",
                   "Content-Type": "application/json"}
        data = {"model": "llama-3.3-70b-versatile", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7}
        waits = [5, 15, 30]
        for attempt in range(1, 5):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://api.groq.com/openai/v1/chat/completions",
                                      headers=headers, json=data,
                                      timeout=aiohttp.ClientTimeout(total=60)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 4:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Groq {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Groq connection failed: {e}")
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception("Groq timed out.")
        raise Exception("Groq unavailable.")

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: OpenRouter
    # ─────────────────────────────────────────────────────────────

    async def _call_openrouter(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json",
            "HTTP-Referer": self.config.get("base_url", "https://kubaik.github.io"),
            "X-Title": self.config.get("site_name", "Tech Blog"),
        }
        data = {
            "model": "google/gemini-flash-1.5", "messages": messages, "max_tokens": max_tokens, "temperature": 0.7,
            "provider": {"ignore": ["Venice"], "allow_fallbacks": True},
        }
        waits = [5, 15, 30]
        for attempt in range(1, 5):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://openrouter.ai/api/v1/chat/completions",
                                      headers=headers, json=data,
                                      timeout=aiohttp.ClientTimeout(total=60)) as r:
                        if r.status == 200:
                            result = await r.json()
                            if "error" in result:
                                raise Exception(
                                    f"OpenRouter error: {result['error']}")
                            return result["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 4:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"OpenRouter {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"OpenRouter connection failed: {e}")
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception("OpenRouter timed out.")
        raise Exception("OpenRouter unavailable.")

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: Cerebras
    # ─────────────────────────────────────────────────────────────

    async def _call_cerebras(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.cerebras_key}",
                   "Content-Type": "application/json"}
        data = {"model": "llama3.1-70b", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7}
        waits = [5, 15, 30]
        for attempt in range(1, 5):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://api.cerebras.ai/v1/chat/completions",
                                      headers=headers, json=data,
                                      timeout=aiohttp.ClientTimeout(total=60)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 4:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Cerebras {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Cerebras connection failed: {e}")
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception("Cerebras timed out.")
        raise Exception("Cerebras unavailable.")

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: Mistral
    # ─────────────────────────────────────────────────────────────

    async def _call_mistral(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.mistral_key}",
                   "Content-Type": "application/json"}
        data = {"model": _MISTRAL_MODEL, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7}
        waits = [_MISTRAL_FREE_TIER_DELAY, 15, 30]
        for attempt in range(1, 5):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(_MISTRAL_API_URL, headers=headers, json=data,
                                      timeout=aiohttp.ClientTimeout(total=60)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 4:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Mistral {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Mistral connection failed: {e}")
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception("Mistral timed out.")
        raise Exception("Mistral unavailable.")

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: NVIDIA NIM
    # ─────────────────────────────────────────────────────────────

    async def _call_nvidia(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.nvidia_key}",
                   "Content-Type": "application/json"}
        data = {"model": _NVIDIA_MODEL, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7, "stream": False}
        waits = [5, 15, 30]
        for attempt in range(1, 5):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(_NVIDIA_API_URL, headers=headers, json=data,
                                      timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 4:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"NVIDIA NIM {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"NVIDIA NIM connection failed: {e}")
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception("NVIDIA NIM timed out.")
        raise Exception("NVIDIA NIM unavailable.")

    # ─────────────────────────────────────────────────────────────
    # PROVIDER: Gemini
    # ─────────────────────────────────────────────────────────────

    async def _call_gemini(self, messages: List[Dict], max_tokens: int) -> str:
        GEMINI_MODEL = "gemini-2.5-flash"
        RETRYABLE = {503, 429, 500, 502, 504}

        try:
            import google.generativeai as genai

            def _sdk_call():
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens, temperature=0.7),
                )
                parts = [("SYSTEM: " if m.get(
                    "role") == "system" else "USER: ") + m.get("content", "") for m in messages]
                return model.generate_content("\n\n".join(parts) + "\n\nASSISTANT:").text
            return await asyncio.get_event_loop().run_in_executor(None, _sdk_call)
        except ImportError:
            pass

        api_url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
        system_parts = [m["content"]
                        for m in messages if m.get("role") == "system"]
        user_parts = [m["content"]
                      for m in messages if m.get("role") != "system"]
        first_user = ("\n\n".join(system_parts) + "\n\n" if system_parts else "") + \
            (user_parts[0] if user_parts else "")
        contents = [{"role": "user", "parts": [{"text": first_user}]}]
        for extra in user_parts[1:]:
            contents.append({"role": "user", "parts": [{"text": extra}]})
        payload = {"contents": contents, "generationConfig": {
            "maxOutputTokens": max_tokens, "temperature": 0.7}}
        waits = [5, 15, 30, 60]
        for attempt in range(1, 6):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            result = await r.json()
                            try:
                                return result["candidates"][0]["content"]["parts"][0]["text"]
                            except (KeyError, IndexError) as e:
                                raise Exception(f"Gemini parse error: {e}")
                        if r.status in RETRYABLE and attempt < 5:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Gemini {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 5:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Gemini connection failed: {e}")
            except asyncio.TimeoutError:
                if attempt < 5:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception("Gemini timed out.")
        raise Exception("Gemini unavailable.")

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

            bundle = await self._generate_content_bundle(title, topic, keywords)
            content = bundle["content"].strip()
            meta_description = bundle["meta_description"].strip()
            seo_keywords = [k.strip()
                            for k in bundle["seo_keywords"] if k.strip()]

            if not keywords:
                keywords = seo_keywords

            word_count = _count_words(content)
            print(f"Generated content: {word_count} words")

            if word_count < MIN_WORD_COUNT:
                print(
                    f"Warning: content only {word_count} words (min {MIN_WORD_COUNT}). Expanding...")
                content = await self._expand_content(content, title, topic)
                word_count = _count_words(content)
                print(f"After expansion: {word_count} words")

            # ── CHANGE: second expansion pass if still short ──────────────
            if word_count < MIN_WORD_COUNT:
                print(
                    f"Still short ({word_count} words). Running second expansion...")
                content = await self._expand_content(content, title, topic)
                word_count = _count_words(content)
                print(f"After second expansion: {word_count} words")

            slug = self._create_slug(title)

            post = BlogPost(
                title=title.strip(),
                content=content,
                slug=slug,
                tags=seo_keywords[:5],
                meta_description=meta_description,
                featured_image=f"/static/images/{slug}.jpg",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                seo_keywords=seo_keywords,
                affiliate_links=[],
                monetization_data={},
            )

            enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
                post.content, topic)
            post.content = enhanced_content
            post.affiliate_links = affiliate_links
            post.monetization_data = self.monetization.generate_ad_slots(
                enhanced_content)

            print("Deriving hashtags from title + keywords (tiered system)...")
            hashtags = _derive_hashtags_from_keywords(
                seo_keywords, topic=topic, title=title, max_hashtags=5
            )
            print(f"Hashtags selected: {', '.join(hashtags)}")

            post.tags = list(set(post.tags + hashtags))[:15]
            post.seo_keywords = list(set(post.seo_keywords + hashtags))[:15]
            post.twitter_hashtags = " ".join(
                f"#{h.replace(' ', '').replace('-', '')}" for h in hashtags
            )

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
                                     max_attempts: int = 2) -> str:
        for attempt in range(1, max_attempts + 1):
            extra_instruction = ""
            if attempt > 1:
                extra_instruction = (
                    " IMPORTANT: Do NOT produce a title similar to any of these: "
                    + ", ".join(f'"{t}"' for t in existing_titles[:10])
                    + ". Choose a clearly different angle."
                )
            title = await self._generate_title(topic, keywords, extra_instruction=extra_instruction)
            title = title.strip().strip('"')
            if not existing_titles:
                return title
            is_dup, match, score = _is_duplicate_title(title, existing_titles)
            if not is_dup:
                print(
                    f"Title accepted (similarity {score:.0%} to nearest existing): {title}")
                return title
            print(
                f"Attempt {attempt}: title too similar ({score:.0%}) to '{match}'. Retrying…")
        return f"{title}"

    async def _generate_title(self, topic: str, keywords: List[str] = None,
                              extra_instruction: str = "") -> str:
        keyword_text = f" Focus on keywords: {', '.join(keywords)}" if keywords else ""
        messages = [
            {"role": "system", "content": "You are a skilled blog title writer. Create engaging, SEO-friendly titles."},
            {"role": "user", "content": (
                f"Generate a compelling blog post title about '{topic}'.{keyword_text} "
                "The title should be catchy, informative, and under 60 characters. "
                "Avoid generic titles starting with 'The Ultimate Guide' or 'Everything You Need to Know'. "
                "Respond with ONLY the title — no quotes, no numbering, no explanation."
                f"{extra_instruction}"
            )},
        ]
        title = await self._call_api_with_fallback(messages, max_tokens=80)
        return title.strip().splitlines()[0].strip().strip('"')

    async def _generate_content_bundle(self, title: str, topic: str, keywords: List[str] = None) -> Dict:
        """Single API call: content + meta_description + seo_keywords as JSON."""
        keyword_text = f"\nKeywords to incorporate naturally: {', '.join(keywords)}" if keywords else ""

        # ── CHANGE: system prompt now requires original opinion/insight ──
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced tech professional with 10+ years of hands-on experience. "
                    "Write in a direct, opinionated voice — share specific insights, real tradeoffs, and concrete numbers. "
                    "Never use filler phrases like 'in today's fast-paced world', 'crucial aspect', or 'it is important to note'. "
                    "Every paragraph must deliver concrete value. Be specific: name actual tools, libraries, companies, and version numbers. "
                    "Take clear stances. Acknowledge tradeoffs honestly. "
                    "IMPORTANT: Every article must include at least one original opinion or counterintuitive insight "
                    "not commonly found in documentation or generic blog posts. "
                    "Draw on real-world production experience, not just theory. "
                    "You MUST respond with ONLY a valid JSON object — no markdown fences, no preamble, no trailing commentary."
                ),
            },
            {
                "role": "user",
                # ── CHANGE: target 2500 words; each section now 200+ words ──
                "content": f"""Write a 2500-word technical blog post titled: "{title}"

Topic: {topic}{keyword_text}

Respond with ONLY a JSON object in this exact shape:
{{
  "content": "<full markdown article body — no title line>",
  "meta_description": "<under 155 chars, specific, no 'Learn how to' opener>",
  "seo_keywords": ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6", "kw7", "kw8"]
}}

Article structure (use exactly these ## headings inside "content"):
## The Problem Most Developers Miss
## How [Topic] Actually Works Under the Hood
## Step-by-Step Implementation
## Real-World Performance Numbers
## Common Mistakes and How to Avoid Them
## Tools and Libraries Worth Using
## When Not to Use This Approach
## My Take: What Nobody Else Is Saying
## Conclusion and Next Steps

Requirements for "content":
- Markdown format
- At least 2 realistic code examples (with language tag, e.g. ```python)
- Specific tool names with version numbers where relevant
- At least 3 concrete numbers (benchmarks, percentages, file sizes, latency figures, etc.)
- Each section minimum 200 words
- "When Not to Use This Approach" must be honest and specific (name real scenarios)
- "My Take: What Nobody Else Is Saying" must contain a genuine, opinionated stance the author holds based on production experience — not a summary of what others say
- Do NOT include the title as a # heading

Requirements for "seo_keywords": 8 items — 2 short-tail, 4 long-tail, 2 question-based.
Avoid in content: vague phrases, padding, lists over 6 items, "dive into", "delve into", "In conclusion", "Overall".
Return ONLY the JSON object.""",
            },
        ]

        raw = await self._call_api_with_fallback(messages, max_tokens=8000)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

        def _sanitize(s):
            result, in_str, esc = [], False, False
            for ch in s:
                if esc:
                    result.append(ch)
                    esc = False
                    continue
                if ch == '\\':
                    result.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    result.append(ch)
                    continue
                if in_str:
                    if ch == '\n':
                        result.append('\\n')
                    elif ch == '\r':
                        result.append('\\r')
                    elif ch == '\t':
                        result.append('\\t')
                    elif ord(ch) < 0x20:
                        result.append(f'\\u{ord(ch):04x}')
                    else:
                        result.append(ch)
                else:
                    result.append(ch)
            return ''.join(result)

        def _repair(text):
            text = text.rstrip()
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
            in_str, esc, depth = False, False, 0
            for ch in text:
                if esc:
                    esc = False
                    continue
                if ch == '\\' and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
            if not in_str and depth == 0:
                return _sanitize(text)
            rep = text
            if in_str:
                rep += '"'
            for _ in range(max(0, rep.count('[') - rep.count(']'))):
                rep += ']'
            for _ in range(max(0, rep.count('{') - rep.count('}'))):
                rep += '}'
            return rep

        def _partial(text):
            data = {}
            m = re.search(
                r'"content"\s*:\s*"(.*?)(?:"\s*,\s*"(?:meta_description|seo_keywords)|"\s*\})', text, re.DOTALL)
            if not m:
                m = re.search(r'"content"\s*:\s*"(.*)', text, re.DOTALL)
            if m:
                data['content'] = m.group(1).replace(
                    '\\n', '\n').replace('\\"', '"').replace('\\t', '\t')
            m = re.search(
                r'"meta_description"\s*:\s*"(.*?)(?:"\s*,\s*"|\"\s*\})', text, re.DOTALL)
            if m:
                data['meta_description'] = m.group(
                    1).replace('\\n', ' ').strip()
            m = re.search(r'"seo_keywords"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if m:
                data['seo_keywords'] = [k.strip().strip('"')
                                        for k in m.group(1).split(',') if k.strip().strip('"')]
            return data

        def _parse(text):
            for attempt in [
                lambda t: json.loads(t),
                lambda t: json.loads(_sanitize(t)),
                lambda t: json.loads(_sanitize(re.search(r'\{.*\}', t, re.DOTALL).group())) if re.search(
                    r'\{.*\}', t, re.DOTALL) else (_ for _ in ()).throw(ValueError()),
                lambda t: json.loads(_sanitize(_repair(t))),
            ]:
                try:
                    return attempt(text)
                except Exception:
                    pass
            print("Warning: JSON unrecoverable — extracting fields individually.")
            data = _partial(text)
            if 'content' in data:
                data.setdefault('meta_description', '')
                data.setdefault('seo_keywords', [])
                return data
            raise ValueError(
                f"Model did not return valid JSON.\nRaw (first 400):\n{text[:400]}")

        data = _parse(raw)
        for key in ("content", "meta_description", "seo_keywords"):
            if key not in data:
                raise ValueError(f"Bundle response missing key: '{key}'")
        return data

    async def _expand_content(self, existing_content: str, title: str, topic: str) -> str:
        messages = [
            {"role": "system", "content": "You are a technical writer expanding existing blog content."},
            {"role": "user", "content": (
                f"The following blog post about '{topic}' is too short. "
                "Add 3 additional detailed sections at the end (each 250+ words) covering:\n"
                "1. Advanced configuration and real edge cases you have personally encountered\n"
                "2. Integration with popular existing tools or workflows, with a concrete example\n"
                "3. A realistic case study or before/after comparison with actual numbers\n\n"
                f"Existing content:\n{existing_content}\n\n"
                "Return the complete article including original content plus the new sections. "
                "Do not include the title line. Be specific — name tools, versions, and metrics."
            )},
        ]
        return await self._call_api_with_fallback(messages, max_tokens=5000)

    # ─────────────────────────────────────────────────────────────
    # THREAD TWEETS
    # ─────────────────────────────────────────────────────────────

    def _build_post_url(self, post_url: str, position: int, style: str) -> str:
        return f"{post_url}?utm_source=twitter&utm_medium=thread&utm_campaign=tweet_{position}&utm_content={style}"

    def _build_thread_tweets(self, post) -> List[str]:
        base_url = self.config.get('base_url', 'https://kubaik.github.io')
        post_url = f"{base_url}/{post.slug}"
        short_title = post.title if len(
            post.title) <= 60 else post.title[:57] + "..."

        if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags:
            hashtags = post.twitter_hashtags
        elif hasattr(post, 'tags') and post.tags:
            clean_tags = [t.replace(' ', '').replace(
                '-', '') for t in post.tags if t and len(t.replace(' ', '').replace('-', '')) >= 2]
            hashtags = " ".join(f"#{t}" for t in clean_tags[:5])
        else:
            hashtags = ""

        topic_phrase = _extract_topic_phrase(post.title, max_words=3)
        topic_words = [w for w in post.title.split() if w.lower()
                       not in _HOOK_STOP_WORDS and len(w) >= 2]
        topic_b = " ".join(topic_words[1:3]) if len(
            topic_words) > 1 else "the fundamentals"
        topic_c = topic_words[-1] if topic_words else "performance"

        hook_style = self.config.get('hook_style', 'knowledge_gap')

        hook_templates = {
            'knowledge_gap': (
                f"🧵 Most people approach {topic_phrase} backwards.\n\n"
                f"They spend weeks on the wrong layer and wonder why nothing scales.\n\n"
                f"The fix is simpler than you think — but only if you understand what's actually going wrong first."
            ),
            'contrarian': (
                f"🧵 Hot take: most {topic_phrase} advice actively makes your system worse.\n\n"
                f"Not because it's wrong in theory — because it ignores what breaks in production.\n\n"
                f"Here's what actually works."
            ),
            'specific_number': (
                f"🧵 3 {topic_phrase} mistakes I see constantly — even from senior engineers:\n\n"
                f"① Skipping the boring foundation work\n"
                f"② Optimising before measuring\n"
                f"③ Ignoring failure modes until they're on fire\n\n"
                f"The third one is the silent killer."
            ),
            'pattern_interrupt': (
                f"🧵 You can tell in 5 minutes whether someone truly understands {topic_phrase} — or just thinks they do.\n\n"
                f"The difference isn't knowledge. It's what they check first when something breaks."
            ),
        }

        hook = hook_templates.get(hook_style, hook_templates['knowledge_gap'])
        url_t2 = self._build_post_url(post_url, position=2, style=hook_style)
        url_t4 = self._build_post_url(post_url, position=4, style=hook_style)

        description = post.meta_description[:150].rstrip()
        if len(post.meta_description) > 150:
            description += "…"

        tweets = [
            hook,
            (
                f"Full breakdown 👇\n"
                f"{description}\n"
                f"What's inside:\n"
                f"→ Why {topic_phrase} fails at scale\n"
                f"→ {topic_b} patterns that actually work\n"
                f"→ Real benchmarks + code\n\n"
                f"{url_t2}"
                + (f"\n\n{hashtags}" if hashtags else "")
            ),
            (
                f"2/ What practitioners do differently:\n\n"
                f"✅ Nail {topic_phrase} fundamentals first\n"
                f"✅ Apply {topic_b} discipline early\n"
                f"✅ Optimise {topic_c} before it gets expensive to fix"
            ),
            (
                f"3/ TL;DR — if you only read one thing on {topic_phrase} this week, make it this.\n\n"
                f"Examples, code, and the mistakes to avoid 👇\n{url_t4}"
            ),
        ]

        return [t if len(t) <= 280 else t[:277] + "..." for t in tweets]

    # ─────────────────────────────────────────────────────────────
    # LOCAL FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _generate_fallback_post(self, topic: str) -> BlogPost:
        """
        CHANGE: fallback now sets used_fallback=True in monetization_data so
        audit_posts() can detect and flag these posts before reapplying to AdSense.
        """
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

## Step-by-Step Implementation

```python
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class {topic_slug}Client:
    def __init__(self, config: Dict[str, Any]):
        self.config      = config
        self.max_retries = config.get("max_retries", 3)
        self.timeout     = config.get("timeout_seconds", 5.0)
        self._connection = None

    def connect(self) -> bool:
        for attempt in range(self.max_retries):
            try:
                self._connection = self._create_connection()
                logger.info(f"Connected on attempt {{attempt + 1}}")
                return True
            except ConnectionError as e:
                wait = 2 ** attempt
                logger.warning(f"Retrying in {{wait}}s: {{e}}")
                time.sleep(wait)
        return False

    def health_check(self) -> bool:
        if not self._connection:
            return False
        try:
            return self._ping()
        except Exception:
            self._connection = None
            return False
```

Step 1: Set environment variables — never hardcode credentials.
Step 2: Start with conservative timeouts (5s) and tune from p99 measurements.
Step 3: Add circuit breaker — stop after 5 failures, wait 30 seconds.
Step 4: Instrument connection count, success rate, p50/p95/p99 latency, error types.
Step 5: Load test with realistic traffic before going live.

## Real-World Performance Numbers

- **Under 1,000 req/min:** Default config works. Focus on correctness.
- **1,000–50,000 req/min:** Connection pooling is critical. Without it, expect 20–35% latency increase.
- **50,000+ req/min:** Single-node coordinators bottleneck. 40% throughput gain moving to clustered setup.

## Common Mistakes and How to Avoid Them

**No timeout:** Set connection timeout (2–5s) and per-operation timeout (1–5s) explicitly.
**Binary error handling:** Connection refused ≠ timeout ≠ auth error. Handle each separately.
**No pool monitoring:** Alert when wait time exceeds 500ms.
**Happy-path-only testing:** Use fault injection in staging.
**DNS caching in containers:** Set TTL to 30–60 seconds.

## Tools and Libraries Worth Using

- **Prometheus + Grafana** for metrics (use histograms, not averages)
- **OpenTelemetry** for distributed tracing (~1–2% overhead)
- **Testcontainers** for real infrastructure in tests
- **k6 or Locust** for load testing
- **tenacity / resilience4j / polly** for circuit breaker and retry

## When Not to Use This Approach

Skip it for low, predictable traffic (under 100 req/min). Skip it without observability — you can't debug what you can't see. Skip it if your team doesn't understand the failure modes; a simpler system they know beats a sophisticated one they don't.

## My Take: What Nobody Else Is Saying

Most guides tell you to add {topic} and call it done. In practice, the hardest part is not the setup — it's the operational burden. Every abstraction you add is a thing your team needs to understand at 2am when it breaks. Start simpler than you think you need to, instrument everything from day one, and only add complexity when metrics prove you need it.

## Conclusion and Next Steps

Production-ready {topic} comes down to systematic failure handling. Add explicit timeouts today. Set up latency histograms this week. Run a chaos test against staging this month."""

        fallback_hashtags = _derive_hashtags_from_keywords(
            [topic_lower, f"{topic_lower} tutorial",
                f"{topic_lower} best practices"],
            topic=topic, title=title, max_hashtags=5,
        )

        post = BlogPost(
            title=title, content=content, slug=slug,
            tags=[topic_lower.replace(
                ' ', '-'), 'development', 'technical-guide', 'best-practices'] + fallback_hashtags,
            meta_description=f"A practical guide to {topic} covering implementation, real performance benchmarks, common mistakes, and honest tradeoffs.",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            seo_keywords=[topic_lower, f"{topic_lower} tutorial", f"{topic_lower} best practices",
                          f"how to use {topic_lower}", f"{topic_lower} performance"],
            affiliate_links=[], monetization_data={},
        )

        # ── CHANGE: flag fallback posts so audit_posts() can detect them ──
        post.monetization_data["used_fallback"] = True

        post.twitter_hashtags = " ".join(f"#{h}" for h in fallback_hashtags)

        enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
            post.content, topic)
        post.content = enhanced_content
        post.affiliate_links = affiliate_links
        post.monetization_data.update(
            self.monetization.generate_ad_slots(enhanced_content))
        post.monetization_data["used_fallback"] = True  # preserve after update
        return post

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _create_slug(self, title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        return slug.strip('-')[:50]

    def save_post(self, post):
        word_count = len(post.content.split())
        if word_count < MIN_WORD_COUNT:
            print(
                f"Warning: saving post with only {word_count} words (min recommended: {MIN_WORD_COUNT})")

        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        post_data = post.to_dict()
        if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags:
            post_data['twitter_hashtags'] = post.twitter_hashtags

        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=2, ensure_ascii=False)

        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")

        print(f"Saved post: {post.title} ({post.slug}) — {word_count} words")
        if post.affiliate_links:
            print(f"  - {len(post.affiliate_links)} affiliate links added")
        print(
            f"  - {post.monetization_data.get('ad_slots', 0)} ad slots configured")
        if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags:
            print(f"  - Twitter hashtags: {post.twitter_hashtags}")


# ─────────────────────────────────────────────────────────────────
# TOPIC PICKER
# ─────────────────────────────────────────────────────────────────

def pick_next_topic(config_path="config.yaml", history_file=".used_topics.json") -> str:
    print(f"Picking topic from {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file {config_path} not found. Run 'python blog_system.py init' first.")

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
        safe_available, skipped = [], []
        for candidate in available:
            is_dup, match, score = _is_duplicate_title(
                candidate, existing_titles, threshold=DUPLICATE_TITLE_THRESHOLD)
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
            print("All available topics covered. Resetting.")
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
        "hook_style": "knowledge_gap",
        "social_accounts": {"twitter": "@KubaiKevin", "linkedin": "your-linkedin-page", "facebook": "your-facebook-page"},
        "content_topics": [
            "How AI Is Changing Everyday Life in 2026", "ChatGPT vs Claude vs Gemini: Which AI Actually Wins",
            "10 AI Tools That Replace Expensive Software", "AI Prompting Secrets Most People Never Learn",
            "How to Use AI to Make Money Online", "AI Tools for Students: Study Smarter Not Harder",
            "Free AI Tools That Professionals Actually Use", "How Companies Are Using AI to Cut Costs",
            "AI-Generated Content: What's Real and What's Fake", "The AI Skills That Will Get You Hired in 2026",
            "How to Build an AI-Powered Side Hustle", "AI vs Human: Where Machines Still Fail",
            "The Hidden Dangers of Relying on AI", "How Hospitals Are Using AI to Save Lives",
            "AI in Education: The Future of Learning", "Tech Salaries in 2026: Who Earns What",
            "How to Get a $150K Tech Job Without a Degree", "Freelance Developer Income: Realistic Numbers",
            "The Tech Skills That Pay the Most Right Now", "How to Negotiate a Tech Salary (Scripts That Work)",
            "Remote Tech Jobs: Where to Find Them in 2026", "Tech Career Roadmap: From Zero to Employed in 12 Months",
            "Why Senior Developers Leave Big Tech Companies", "The Highest-Paying Programming Languages in 2026",
            "How to Build Passive Income as a Developer", "Breaking Into Tech at 30, 40, or 50",
            "Tech Interview Red Flags That Cost Candidates Jobs", "The Fastest Growing Tech Roles Right Now",
            "Why Developers Burn Out and How to Prevent It", "How to Get Promoted Faster in Tech",
            "How to Build a SaaS Product as a Solo Developer", "The Tech Stack for Bootstrapped Startups in 2026",
            "How Indie Hackers Are Making $10K/Month", "No-Code vs Code: When to Use Each",
            "From Idea to Launch: Building an MVP in 30 Days", "How to Validate a Startup Idea Before Building",
            "The Cheapest Way to Deploy a Web App in 2026", "Why Most Side Projects Fail (And How to Fix It)",
            "How to Find Your First 100 Customers as a Developer", "Building in Public: What Works and What Doesn't",
            "Open Source Projects That Made Millions", "How to Price Your SaaS Product",
            "What VCs Actually Look for in Tech Startups", "Startup Failure Lessons from Founders Who Lost Everything",
            "The Solo Developer's Guide to Scaling", "The AI Workflow That Saves 10 Hours a Week",
            "Best AI Coding Assistants Compared: Copilot vs Cursor vs Others",
            "How to Use AI for Content Creation (Without It Sounding Robotic)",
            "AI for Data Analysis: No Coding Required", "The Best AI Image Tools in 2026",
            "How Developers Are Using AI to 10x Their Output", "Automating Your Life with AI and Python",
            "AI Tools That Write Better Code Than Most Juniors", "How to Build a Personal AI Assistant",
            "The Prompt Engineering Techniques That Actually Work", "AI Agents: What They Are and Why They Matter",
            "How to Use AI to Learn Any Skill Faster", "The Best Free AI APIs for Developers",
            "Building AI-Powered Apps Without Machine Learning Knowledge", "AI for Personal Finance: Tools and Strategies",
            "Will AI Replace Software Developers? The Honest Answer", "Remote Work vs Office: What the Data Actually Shows",
            "The 4-Day Work Week in Tech: Companies Trying It", "Why So Many Developers Are Leaving Big Tech",
            "Tech Layoffs 2026: What's Really Happening", "The Skills That Will Be Worthless in 5 Years",
            "How Social Media Algorithms Work (And How to Beat Them)", "The Dark Side of Big Tech: What Insiders Say",
            "Why Everyone Should Learn to Code (And Why That's Wrong)", "How AI Is Making Wealth Inequality Worse",
            "The Countries Winning the AI Race", "Digital Nomad Life: The Real Costs Nobody Mentions",
            "Why Junior Developer Jobs Are Disappearing", "The Tech Industry's Mental Health Crisis",
            "How Technology Is Changing Human Relationships", "Why Most Digital Transformations Fail",
            "How Netflix Decides What to Build Next", "The Tech Behind Amazon's One-Click Empire",
            "How Stripe Became the Internet's Payment System", "Why WhatsApp Was Worth $19 Billion",
            "The Engineering Culture That Built Google", "How Apple Maintains Premium Pricing in a Competitive Market",
            "The Lessons Startups Learn Too Late", "How Big Tech Makes Money From Your Data",
            "The Open Source Business Models That Work", "Why Slack Failed to Beat Microsoft Teams",
            "How Figma Was Built and Why Adobe Tried to Buy It", "The API Economy: How Twilio and Stripe Print Money",
            "Why Most Tech Companies Never Become Profitable", "Platform Business Models Explained",
            "Generative AI and Large Language Models Explained", "How Neural Networks Actually Learn",
            "Prompt Engineering for Real-World Applications", "Vector Databases and Embeddings: A Practical Guide",
            "Building AI Agents That Actually Work", "MLOps: Deploying AI Models Without Breaking Everything",
            "Fine-Tuning LLMs Without a GPU", "AI Model Monitoring: Catching Drift Before It Hurts",
            "Retrieval-Augmented Generation (RAG) Explained Simply", "Multi-Modal AI: When Models See, Hear, and Read",
            "AI Ethics: The Problems Big Tech Doesn't Want to Discuss", "Federated Learning: AI That Respects Privacy",
            "AI for Time Series Forecasting in Practice", "Explainable AI: Making Black Boxes Transparent",
            "Computer Vision Applications in the Real World", "How Hackers Actually Break Into Systems",
            "The Biggest Data Breaches of 2026 and What Went Wrong", "Zero Trust Security: Why Perimeter Defense Is Dead",
            "How to Protect Your Personal Data from Corporations", "Password Security: What Actually Works in 2026",
            "How Ransomware Attacks Work and How to Survive One", "API Security Mistakes That Got Companies Hacked",
            "OAuth 2.0 and JWT Authentication Deep Dive", "Penetration Testing: How Ethical Hackers Think",
            "Social Engineering: The Human Side of Cybersecurity", "Cloud Security Best Practices for Developers",
            "How to Run a Security Audit on Your Web App", "DDoS Attacks: How They Work and How to Stop Them",
            "Container Security in Production Kubernetes", "The Security Vulnerabilities in Most Mobile Apps",
            "React vs Next.js vs Remix: Choosing the Right Tool", "Full-Stack Development in 2026: The Best Stack",
            "Building Real-Time Apps with WebSockets", "Web Performance: Why Your Site Loads Slow and How to Fix It",
            "TypeScript Patterns That Actually Save Time", "CSS Tricks That Senior Developers Actually Use",
            "Building a Production-Ready App with Supabase", "The Jamstack in 2026: Still Worth It?",
            "Web Accessibility: The Laws and the Code", "GraphQL vs REST vs tRPC in 2026",
            "Progressive Web Apps: When They Beat Native", "Server Components vs Client Components Explained",
            "WebAssembly: When JavaScript Isn't Fast Enough", "Building a Micro-Frontend Architecture",
            "Modern Authentication Patterns for Web Apps", "System Design Interview: How to Think Like a Senior Engineer",
            "How Netflix Handles 200 Million Concurrent Streams", "Designing a URL Shortener That Handles Billions of Requests",
            "Microservices vs Monolith: The Honest Comparison", "Event-Driven Architecture in Practice",
            "Database Indexing: The Hidden Performance Secret", "Redis in Production: Patterns That Scale",
            "Building an API That Can Handle a Million Requests", "The CAP Theorem Explained Simply",
            "How Stripe Processes Payments Without Losing Data", "Designing a Chat System Like WhatsApp",
            "Rate Limiting Strategies That Actually Work", "Message Queues: Kafka vs RabbitMQ vs SQS",
            "Database Sharding: When and How to Do It", "Serverless Architecture: Real Costs and Real Limits",
            "The DevOps Mistakes That Cause Outages", "Kubernetes: When It Helps and When It Hurts",
            "Docker in Production: What No One Tells You", "CI/CD Pipelines That Actually Prevent Bugs",
            "Cloud Cost Optimization: Cutting Your AWS Bill in Half", "Infrastructure as Code with Terraform: The Real Guide",
            "GitOps: The Deployment Strategy Worth Understanding", "Monitoring Your Application Before Users Complain",
            "The On-Call Engineer's Survival Guide", "Blue-Green vs Canary Deployments: Which and When",
            "Log Management at Scale: What Works", "Site Reliability Engineering Principles That Matter",
            "Multi-Cloud Strategy: Smart or Overkill", "Secrets Management: Keeping Credentials Safe",
            "Platform Engineering: Building Internal Developer Platforms",
            "Building Your First Data Pipeline That Doesn't Break", "Apache Kafka for Developers Who Aren't Data Engineers",
            "Data Warehouse vs Data Lake vs Lakehouse: Which One", "Real-Time Data Processing at Scale",
            "Data Quality: Why Your Analytics Are Lying to You", "Apache Spark Without the Headaches",
            "Snowflake vs BigQuery vs Redshift in 2026", "A/B Testing: How to Run Experiments That Mean Something",
            "Data Mesh Architecture Explained", "Business Intelligence Tools for Engineering Teams",
            "React Native vs Flutter in 2026: Final Answer", "Building a Mobile App That Users Don't Delete",
            "Mobile Performance: Why Your App Feels Slow", "Push Notifications Done Right",
            "App Store Optimization: What Actually Moves Rankings", "Swift for iOS: Patterns That Scale",
            "Kotlin for Android: Modern Development Guide", "Mobile Security Vulnerabilities and Fixes",
            "Cross-Platform vs Native: The Real Trade-offs", "Mobile CI/CD Automation in Practice",
            "Blockchain Beyond the Hype: Real Use Cases", "Web3 Development: What It Actually Takes",
            "IoT Architecture for Developers", "Edge Computing: Why It Matters for Your App",
            "Quantum Computing for Software Engineers", "AR Development with Apple Vision Pro",
            "Digital Twin Technology in Industry", "5G's Real Impact on Application Development",
            "Low-Code Platforms: Threat or Tool for Developers", "Robotics Process Automation in Enterprise",
            "Clean Code: The Rules That Actually Matter", "SOLID Principles Applied to Real Projects",
            "Test-Driven Development That Doesn't Slow You Down", "Code Review: How to Give Feedback That Improves Code",
            "Refactoring Legacy Code Without Breaking Everything", "Technical Debt: How to Measure and Pay It Down",
            "Design Patterns You'll Actually Use", "Documentation That Developers Actually Read",
            "Agile in Practice: What Works, What's Theatre", "Pair Programming: When It's Worth It",
            "How Senior Developers Think Differently", "The Mental Models Every Developer Needs",
            "How to Learn a New Programming Language Fast", "Debugging Mindset: How Experts Find Bugs",
            "How to Read Other People's Code Effectively", "Technical Writing for Developers",
            "Building a Second Brain as a Developer", "How to Contribute to Open Source Projects",
            "Developer Productivity: What Research Actually Shows", "Managing Up: How Developers Build Influence",
            "Python in 2026: What's New and What Changed", "JavaScript Features That Changed How We Code",
            "TypeScript Advanced Patterns Worth Learning", "Go Concurrency: Why Gophers Love Goroutines",
            "Rust for Developers Coming from Python or JavaScript", "Java in 2026: Still Relevant or Time to Move On",
            "Functional Programming Concepts in Practical Code", "SQL Tricks That Replace Complex Application Code",
            "Bash Scripting for Developers Who Avoid the Terminal", "Python vs Go vs Rust: Choosing for Your Use Case",
            "VS Code Setup That Makes You 2x Faster", "Git Commands That Senior Developers Use Daily",
            "Terminal Productivity for Developers", "Debugging Techniques That Find Bugs in Minutes",
            "API Testing: Beyond Basic Postman Requests", "Database Tools Worth Having in 2026",
            "The Developer's Guide to Time Management", "Automating Repetitive Dev Tasks with Python",
            "Command Line Tools Every Developer Should Know", "Building a Development Environment That Doesn't Frustrate",
            "Application Performance Monitoring That Prevents Incidents",
            "Database Query Optimization: Finding and Fixing Slow Queries",
            "Frontend Performance: Core Web Vitals Explained", "Image Optimization for Web in 2026",
            "Lazy Loading and Code Splitting in Practice", "Memory Leaks: How to Find and Fix Them",
            "Profiling Python Applications for Speed", "Network Performance Optimization for APIs",
            "Algorithm Optimization: Practical Big O Analysis", "Caching Strategies That Actually Improve Performance",
        ],
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("Created sample config.yaml")
    print("\nAdd GitHub secrets: GROQ_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY, MISTRAL_API_KEY, NVIDIA_API_KEY, GEMINI_API_KEY")


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
            print(
                "Done! API chain: Groq → OpenRouter → Cerebras → Mistral → NVIDIA NIM → Gemini → local template")

        elif mode == "auto":
            print("Starting automated blog generation...")
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

                print(f"\nPost '{blog_post.title}' generated successfully!")
                print(f"Twitter hashtags: {blog_post.twitter_hashtags}")

                visibility = VisibilityAutomator(config)
                print("\nPosting as thread for maximum impressions...")
                thread_result = visibility.post_thread(blog_post)

                if thread_result['success']:
                    print(
                        f"Thread posted ({thread_result['tweet_count']} tweets)")
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
                                'twitter', '@KubaiKevin')
                            hashtags = getattr(blog_post, 'twitter_hashtags', '') or " ".join(
                                f"#{t.replace(' ','').replace('-','')}" for t in sorted(getattr(blog_post, 'tags', []), key=len)[:3] if t
                            )
                            followup_text = (
                                f"Found this useful? Follow {username} for daily threads on AI, dev tools, and software engineering\n\n{hashtags}"
                            ).strip()[:280]

                            followup_response = visibility.twitter_client.create_tweet(
                                text=followup_text, in_reply_to_tweet_id=last_tweet_id,
                            )
                            followup_id = followup_response.data['id']
                            twitter_user = visibility._username or "KubaiKevin"
                            open(flag_file, 'w').close()
                            print(
                                f"Follow-up reply: https://twitter.com/{twitter_user}/status/{followup_id}")
                        else:
                            print("Follow-up reply already posted today, skipping.")

                    except Exception as followup_err:
                        print(
                            f"Follow-up reply failed (skipping): {followup_err}")

                else:
                    print(
                        f"Thread failed ({thread_result.get('error')}). Falling back to single tweet...")
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
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            generator = StaticSiteGenerator(BlogSystem(config))
            generator.generate_site()
            print("Site rebuilt successfully!")

        elif mode == "cleanup":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            blog_system.cleanup_posts()
            StaticSiteGenerator(blog_system).generate_site()
            print("Cleanup and rebuild complete!")

        # ── CHANGE: new 'audit' CLI command ───────────────────────────────
        elif mode == "audit":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            blog_system.purge_low_quality_posts(dry_run=True)

        # ── CHANGE: new 'purge' CLI command (actually deletes) ────────────
        elif mode == "purge":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            blog_system.purge_low_quality_posts(dry_run=False)
            StaticSiteGenerator(blog_system).generate_site()
            print("Purge and rebuild complete!")

        elif mode == "debug":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            print(
                f"Output directory: {blog_system.output_dir} (exists: {blog_system.output_dir.exists()})")
            if blog_system.output_dir.exists():
                for item in blog_system.output_dir.iterdir():
                    print(
                        f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        for fname in ["post.json", "index.md", "social_posts.json"]:
                            print(
                                f"    {fname}: {'Yes' if (item/fname).exists() else 'No'}")
                        if (item/"post.json").exists():
                            try:
                                with open(item/"post.json") as f:
                                    data = json.load(f)
                                wc = _count_words(data.get('content', ''))
                                is_fb = data.get('monetization_data', {}).get(
                                    'used_fallback', False)
                                print(
                                    f"    Title: {data.get('title','Unknown')} | Words: {wc} {'✓' if wc >= MIN_WORD_COUNT else '⚠'} {'[FALLBACK]' if is_fb else ''}")
                            except Exception as e:
                                print(f"    Invalid JSON: {e}")
            blog_system.cleanup_posts()
            StaticSiteGenerator(blog_system).generate_site()

        elif mode == "social":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            generator = StaticSiteGenerator(blog_system)
            posts = generator._get_all_posts()
            visibility = VisibilityAutomator(config)
            for post in posts:
                social_posts = visibility.generate_social_posts(post)
                with open(blog_system.output_dir / post.slug / "social_posts.json", 'w') as f:
                    json.dump(social_posts, f, indent=2)
                print(f"Social posts generated for: {post.title}")
            print("Done!")

        elif mode == "test-twitter":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            visibility = VisibilityAutomator(config)
            print(f"Connection test: {visibility.test_twitter_connection()}")

        elif mode == "dedup":
            import subprocess
            subprocess.run(["python", "deduplicate_posts.py",
                           "--delete"] + sys.argv[2:])

        else:
            print(
                "Usage: python blog_system.py [init|auto|build|cleanup|audit|purge|debug|social|test-twitter|dedup]")

    else:
        print("AI Blog System — Usage: python blog_system.py [command]")
        print("Commands: init | auto | build | cleanup | audit | purge | debug | social | test-twitter | dedup")
