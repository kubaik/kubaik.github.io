
import os
import json
import random
import re
import yaml
import asyncio
import aiohttp
import requests
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from blog_post import BlogPost
from monetization_manager import MonetizationManager
from seo_optimizer import SEOOptimizer
from visibility_automator import VisibilityAutomator
from static_site_generator import StaticSiteGenerator
from hashtag_manager import HashtagManager, add_hashtags_to_post

from adsense_fixes.internal_linker import build_posts_index, inject_internal_links
# PATCH 2: new AdSense-readiness imports ─────────────────────────────────────
from adsense_fixes.similarity_guard import SimilarityGuard
from adsense_fixes.link_validator import validate_post_links
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# Duplicate-detection helpers
# ─────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is",
    "are", "with", "how", "your", "my", "our", "its", "on", "at",
    "by", "from", "this", "that", "best", "using", "guide", "complete",
    "introduction", "overview", "tutorial", "tips", "top", "ways",
    "ways", "tricks", "steps", "things", "methods", "approach",
    "ace", "pass", "nail", "master", "learn", "know",
    "add", "build", "get", "make", "use", "do",
    "without", "beyond", "instead", "heres", "here",
    "real", "actually", "truly", "really",
    "quick", "fast", "simple", "easy", "practical",
    "vs", "versus",
}

DUPLICATE_TITLE_THRESHOLD = 0.35

MIN_WORD_COUNT = 2000
MIN_WORD_PURGE = 1500

MAX_GENERATION_ATTEMPTS = 5
MIN_ACCEPTABLE_WORDS = 1500

_HASHTAG_MAX_SOURCE_WORDS = 3
_HASHTAG_MAX_CHARS = 24


def _to_single_word_tags(tags: List[str]) -> List[str]:
    result = []
    seen: set = set()

    for tag in tags:
        tag = tag.lstrip('#').strip()
        if not tag:
            continue

        words = [w for w in re.split(r'[\s\-_/]+', tag) if w]

        if len(words) > _HASHTAG_MAX_SOURCE_WORDS:
            continue

        camel = ''.join(w.capitalize() for w in words if w)

        if len(camel) > _HASHTAG_MAX_CHARS:
            continue

        key = camel.lower()
        if camel and key not in seen:
            seen.add(key)
            result.append(camel)

    return result


def _normalise_title(text: str) -> str:
    text = text.lower()
    _VARIANTS = [
        (r'\bpostgresql\b', 'postgres'),
        (r'\bpostgres\b',   'postgres'),
        (r'\bmysql\b',      'sql'),
        (r'\bwebsockets?\b', 'websocket'),
        (r'\breal[\-\s]time\b', 'realtime'),
        (r'\bcs\s*degree\b', 'csdegree'),
        (r'\bno[\-\s]code\b', 'nocode'),
        (r'\bai[\-\s]generated\b', 'aigenerated'),
        (r'\bfull[\-\s]stack\b', 'fullstack'),
        (r'\bback[\-\s]end\b', 'backend'),
        (r'\bfront[\-\s]end\b', 'frontend'),
        (r'\bopen[\-\s]source\b', 'opensource'),
        (r'\b\d+x\b', 'Nx'),
        (r'\b\d+%\b', 'PCT'),
    ]
    for pattern, replacement in _VARIANTS:
        text = re.sub(pattern, replacement, text)
    return text


def _tokenise(text: str) -> set:
    text = _normalise_title(text)
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    tokens = {w for w in words if w not in _STOP_WORDS and len(w) > 2}

    word_list = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    for i in range(len(word_list) - 1):
        tokens.add(f"{word_list[i]}_{word_list[i+1]}")

    return tokens


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    base_score = intersection / union

    overlap_vs_shorter = intersection / min(len(a), len(b))
    return max(base_score, overlap_vs_shorter * 0.7)


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
# Twitter posting flag
# ─────────────────────────────────────────────────────────────────

def _twitter_posting_enabled() -> bool:
    raw = os.getenv("ENABLE_TWITTER_POSTING", "false").strip().lower()
    enabled = raw == "true"
    if not enabled:
        print(
            f"Twitter posting DISABLED "
            f"(ENABLE_TWITTER_POSTING={os.getenv('ENABLE_TWITTER_POSTING', '<not set>')})"
        )
    return enabled


# ─────────────────────────────────────────────────────────────────
# Meta description derivation
# ─────────────────────────────────────────────────────────────────

def _extract_numbers(text: str) -> str:
    patterns = [
        r'\d+\s*%',
        r'\d+x\s+(?:faster|cheaper|more|improvement)',
        r'(?:cut|reduce|save|improve)\w*\s+(?:by\s+)?\d+',
        r'\d+\s*ms',
        r'\d+\s*(?:seconds?|minutes?)\s+(?:faster|saved)',
        r'under\s+\d+\s*ms',
        r'\d+\s*(?:req|requests?)(?:/|\s+per\s+)(?:s|sec|second|min|minute)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 10)
            end = min(len(text), m.end() + 30)
            snippet = text[start:end].strip()
            snippet = re.sub(r'\s+', ' ', snippet)
            snippet = re.sub(r'[,;:\s]+$', '', snippet)
            return snippet
    return ""


def _derive_description(content: str, title: str, max_len: int = 155) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", content)
    text = re.sub(r"`[^`]+`",        " ", text)
    text = re.sub(r"#{1,6}\s+",      " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}",      "",  text)
    text = re.sub(r"\s+",            " ", text).strip()

    _SKIP_PATTERNS = re.compile(
        r'^(I |A colleague|This took me|I\'ve|The short version|I ran into|'
        r'I spent|I have |Here\'s what|Writing this|This is a topic|'
        r'Most of the answers|Most tutorials|I noticed|I found|'
        r'I was surprised|I built|I worked|I saw )',
        re.IGNORECASE
    )

    sentences = re.split(r'(?<=[.!?])\s+', text)

    _NUMBER_RE = re.compile(
        r'\b(\d+\s*%|\d+x\b|\$\d|\d+\s*ms|\d+\s*req|p\d{2}|'
        r'\d+,\d{3}|\d+\s*min\b|\d+\s*sec\b|cut\s+\w+\s+by)',
        re.IGNORECASE
    )
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 40 or len(sent) > max_len * 2:
            continue
        if _SKIP_PATTERNS.match(sent):
            continue
        if _NUMBER_RE.search(sent):
            if len(sent) > max_len:
                sent = sent[:max_len].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
            return sent

    _TOOL_RE = re.compile(
        r'\b(Python|Node\.js|TypeScript|PostgreSQL|Redis|AWS|Lambda|Docker|'
        r'FastAPI|Django|React|Next\.js|Kubernetes|Kafka|MongoDB|MySQL|'
        r'SQLite|Terraform|GitHub|M-Pesa|Paystack|Flutterwave|LLM|GPT|Claude)\b'
    )
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 40:
            continue
        if _SKIP_PATTERNS.match(sent):
            continue
        if _TOOL_RE.search(sent):
            if len(sent) > max_len:
                sent = sent[:max_len].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
            return sent

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 40:
            continue
        if _SKIP_PATTERNS.match(sent):
            continue
        if len(sent) > max_len:
            sent = sent[:max_len].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
        return sent

    keyword = title.replace(":", " —").replace(" vs ", " versus ")
    fallback = f"Practical guide to {keyword} — with code examples and production notes."
    return fallback[:max_len]


def audit_posts(docs_dir: Path) -> Dict:
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
            elif wc < MIN_WORD_PURGE:
                results["short"].append((post_dir.name, wc))
            else:
                results["ok"].append(post_dir.name)
        except Exception as e:
            print(f"Audit error for {post_dir.name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────
# Content quality validation  (PATCH 7 applied here)
# ─────────────────────────────────────────────────────────────────

def _validate_content_quality(content: str, title: str):
    warnings = []
    hard_failures = []
    word_count = len(content.split())

    if word_count < 1500:
        hard_failures.append(
            f"Word count {word_count} is below the absolute minimum of 1500. "
            "Google AdSense reviewers reject thin content immediately."
        )

    boilerplate_markers = [
        "class {topic_slug}Client",
        "class Client:",
        "max_retries = config.get",
        "{topic_slug}",
        "{topic}",
        "topic_slug",
    ]
    for marker in boilerplate_markers:
        if marker in content:
            hard_failures.append(
                f"Template boilerplate detected: '{marker[:40]}'. "
                "This post will be rejected as low-value/AI-generated content."
            )

    title_words = set(re.sub(r'[^\w\s]', '', title.lower()).split())
    title_words.discard('the')
    title_words.discard('a')
    title_words.discard('an')
    if title_words:
        first_para = content[:500]
        first_para_words = set(
            re.sub(r'[^\w\s]', '', first_para.lower()).split())
        title_overlap = len(title_words & first_para_words) / len(title_words)
        if title_overlap > 0.85 and word_count < 2000:
            hard_failures.append(
                f"Opening section appears to be a generic restatement of the title "
                f"({title_overlap:.0%} title word overlap in first 500 chars). "
                "This pattern is flagged as low-value content."
            )

    if word_count < 2000:
        warnings.append(f"Word count low: {word_count} (target ≥ 2000)")

    first_person_re = re.compile(
        r"\b(I |I've |I'm |I found|I ran|I spent|I learned|I noticed|I tested|"
        r"I built|I worked|I saw |I was |I have |I had |I used )\b"
    )
    if not first_person_re.search(content):
        warnings.append(
            "No first-person sentences found. E-E-A-T 'Experience' signal missing."
        )

    # PATCH 7: Code blocks promoted from warning → hard failure
    # Technical posts without code examples are flagged as thin content.
    if "```" not in content:
        hard_failures.append(
            "No code examples found. Technical blog posts must contain at least one "
            "fenced code block. Google's quality raters flag technical posts without "
            "code as low-value content. Regenerate with code examples included."
        )

    # PATCH 7: Concrete numbers promoted from warning → hard failure
    number_re = re.compile(
        r"\b(\d+%|\d+ms|\d+x\b|\$\d|\d+ req|\d+ min|p\d{2}|\d+,\d{3})"
    )
    if not number_re.search(content):
        hard_failures.append(
            "No concrete numbers or metrics found (ms, %, cost, line count, etc.). "
            "Specificity is a core E-E-A-T signal. Regenerate with real benchmarks, "
            "version numbers, or performance figures."
        )

    version_re = re.compile(
        r'\b(Python|Node\.js|TypeScript|PostgreSQL|Redis|Django|FastAPI|React|'
        r'Next\.js|Docker|Kubernetes|Kafka|MySQL)\s+\d+[\.\d]*\b',
        re.IGNORECASE
    )
    if not version_re.search(content):
        warnings.append(
            "No version-pinned tool reference found (e.g. 'Python 3.11', 'Redis 7.2'). "
            "Version pins are a specificity signal that distinguishes original from generic content."
        )

    # PATCH 7: FAQ section promoted from warning → hard failure
    # FAQPage structured data is a meaningful AdSense quality signal.
    if "frequently asked questions" not in content.lower() and "## faq" not in content.lower():
        hard_failures.append(
            "No FAQ section found. A 'Frequently Asked Questions' section is required — "
            "it enables FAQPage structured data, which is a key AdSense quality signal. "
            "The generation prompt already requests this section; check why it was omitted."
        )

    if "|" not in content:
        warnings.append(
            "No markdown table found. A comparison table signals substantive, "
            "structured content — reviewers notice its absence in technical posts."
        )

    # PATCH 7: E-E-A-T author footer promoted from warning → hard failure
    if "### About this article" not in content:
        hard_failures.append(
            "E-E-A-T author footer missing. inject_eeat_signals() must run before "
            "save_post(). Without this section, the post has no authorship signal — "
            "a critical trust requirement for AdSense review."
        )

    filler_phrases = [
        "in today's fast-paced",
        "in the ever-evolving",
        "dive into",
        "delve into",
        "game-changer",
        "it's important to note",
        "needless to say",
        "comprehensive guide",
        "this article will",
        "we will explore",
        "in conclusion",
        "revolutionize",
        "transformative",
        "cutting-edge",
        "state-of-the-art",
        "paradigm shift",
        "harness the power",
        "unlock the potential",
        "as an ai language model",
        "as a large language model",
        "i cannot provide",
        "i don't have access",
        "let's explore",
        "let's dive",
        "look no further",
        "in this blog post",
        "stay tuned",
    ]
    detected = [p for p in filler_phrases if p.lower() in content.lower()]
    if detected:
        warnings.append(
            f"AI-pattern filler phrases detected: {', '.join(repr(p) for p in detected[:4])}"
        )

    title_filler_re = re.compile(
        r"^(a |an |the |complete |ultimate |comprehensive |introduction to |"
        r"guide to |overview of |everything you need)",
        re.IGNORECASE,
    )
    if title_filler_re.match(title.strip()):
        warnings.append(f"Title starts with filler word: '{title[:40]}'")

    if len(title) > 60:
        warnings.append(
            f"Title too long ({len(title)} chars). Target ≤ 60 for SERP display."
        )

    first_200 = content[:200].lower()
    generic_openers = [
        "in this", "today we", "welcome to", "this guide covers",
        "if you're looking", "are you looking", "have you ever",
        "whether you're a beginner", "this post will",
    ]
    for opener in generic_openers:
        if first_200.startswith(opener) or f"\n{opener}" in first_200:
            warnings.append(
                f"Generic opener detected in first 200 chars: '{opener}'. "
                "Start with a specific claim, number, or observation instead."
            )
            break

    return warnings, hard_failures


# ─────────────────────────────────────────────────────────────────
# Topic phrase extractor
# ─────────────────────────────────────────────────────────────────

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
# Tiered hashtag system
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
        "platform eng":    ["PlatformEngineering"],
        "devsecops":       ["DevSecOps"],
        "agentic":         ["AgenticAI"],
        "multi-agent":     ["MultiAgent"],
        "vibe cod":        ["VibeCoding"],
        "claude code":     ["ClaudeCode"],
        "cursor":          ["CursorAI"],
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
        "andela":            ["TechCareer", "AfricaTech"],
        "africa tech":       ["AfricaTech"],
        "nairobi":           ["AfricaTech", "NairobiTech"],
    },
}


def _is_valid_hashtag(tag: str) -> bool:
    if not tag:
        return False
    if not re.match(r'^[A-Za-z0-9]+$', tag):
        return False
    if len(tag) > _HASHTAG_MAX_CHARS:
        return False
    return True


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
                    if _is_valid_hashtag(tag) and tag not in selected[tier]:
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
            words = [w for w in re.split(r'[\s\-_/]+', kw) if w]

            if words and words[0] in question_starters:
                continue
            if len(words) > _HASHTAG_MAX_SOURCE_WORDS:
                continue

            tag = "".join(w.capitalize() for w in words)
            tag = re.sub(r"[^\w]", "", tag)

            if _is_valid_hashtag(tag) and tag not in result:
                result.append(tag)

            if len(result) >= max_hashtags:
                break

    seen: set = set()
    final: List[str] = []
    for tag in result:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            final.append(tag)

    return final[:max_hashtags]


# ─────────────────────────────────────────────────────────────────
# Provider constants
# ─────────────────────────────────────────────────────────────────

_MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
_MISTRAL_FREE_TIER_DELAY = 1.2

_NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"

_GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
_GITHUB_MODEL = "Llama-4-Scout-17B-16E-Instruct"

_CF_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


# ─────────────────────────────────────────────────────────────────
# Eight rotating article structures
# ─────────────────────────────────────────────────────────────────

_STRUCTURE_SETS = [
    (
        "deep_dive",
        [
            "## The gap between what the docs say and what production needs",
            "## How {topic} actually works under the hood",
            "## Step-by-step implementation with real code",
            "## Performance numbers from a live system",
            "## The failure modes nobody warns you about",
            "## Tools and libraries worth your time",
            "## When this approach is the wrong choice",
            "## My honest take after using this in production",
            "## What to do next",
        ],
        (
            "Write like a practitioner explaining to a colleague, not a textbook author. "
            "Include at least one moment where you say what surprised you or contradicted your expectations."
        ),
    ),
    (
        "tutorial",
        [
            "## Why I wrote this (the problem I kept hitting)",
            "## Prerequisites and what you'll build",
            "## Step 1 — set up the environment",
            "## Step 2 — core implementation",
            "## Step 3 — handle edge cases and errors",
            "## Step 4 — add observability and tests",
            "## Real results from running this",
            "## Common questions and variations",
            "## Where to go from here",
        ],
        (
            "Write in tutorial voice — direct, numbered, action-oriented. "
            "Each step should explain WHY before showing HOW. "
            "Include at least one 'gotcha' you discovered while writing or testing this."
        ),
    ),
    (
        "opinion",
        [
            "## The conventional wisdom (and why it's incomplete)",
            "## What actually happens when you follow the standard advice",
            "## A different mental model",
            "## Evidence and examples from real systems",
            "## The cases where the conventional wisdom IS right",
            "## How to decide which approach fits your situation",
            "## Objections I've heard and my responses",
            "## What I'd do differently if starting over",
            "## Summary",
        ],
        (
            "This is an opinion piece. Take a clear, defensible stance in the opening paragraph. "
            "Steelman the opposing view before rebutting it. "
            "Use phrases like 'in my experience', 'I've seen this fail when', 'the honest answer is'. "
            "Avoid hedging — readers come to opinion pieces for conviction."
        ),
    ),
    (
        "comparison",
        [
            "## Why this comparison matters right now",
            "## Option A — how it works and where it shines",
            "## Option B — how it works and where it shines",
            "## Head-to-head: performance",
            "## Head-to-head: developer experience",
            "## Head-to-head: operational cost",
            "## The decision framework I use",
            "## My recommendation (and when to ignore it)",
            "## Final verdict",
        ],
        (
            "Structure this as a genuine comparison, not a sponsored review. "
            "Lead each 'head-to-head' section with a concrete number or test result. "
            "The recommendation must be conditional — 'use X if Y, use Z if W'. "
            "Acknowledge weaknesses in your preferred option."
        ),
    ),
    (
        "case_study",
        [
            "## The situation (what we were trying to solve)",
            "## What we tried first and why it didn't work",
            "## The approach that worked",
            "## Implementation details",
            "## Results — the numbers before and after",
            "## What we'd do differently",
            "## The broader lesson",
            "## How to apply this to your situation",
            "## Resources that helped",
        ],
        (
            "Write this as a narrative — there should be a problem, an attempt, a failure or complication, "
            "and a resolution. Use past tense for the story sections. "
            "Every claim about results must include a number (latency, cost, lines of code, time saved, etc.). "
            "The 'broader lesson' section is where you zoom out — make it a principle, not just a summary."
        ),
    ),
    (
        "explainer",
        [
            "## The one-paragraph version (read this first)",
            "## Why this concept confuses people",
            "## The mental model that makes it click",
            "## A concrete worked example",
            "## How this connects to things you already know",
            "## Common misconceptions, corrected",
            "## The advanced version (once the basics are solid)",
            "## Quick reference",
            "## Further reading worth your time",
        ],
        (
            "Start with the simplest possible accurate explanation. Build complexity gradually. "
            "Use analogies freely — name them as analogies ('think of it like a...') so they don't mislead. "
            "The 'quick reference' section should be a scannable table or bullet list a reader can bookmark."
        ),
    ),
    (
        "listicle",
        [
            "## Why this list exists (what I was actually trying to solve)",
            "## How I evaluated each option",
            "## {topic} — the full ranked list",
            "## The top pick and why it won",
            "## Honorable mentions worth knowing about",
            "## The ones I tried and dropped (and why)",
            "## How to choose based on your situation",
            "## Frequently asked questions",
            "## Final recommendation",
        ],
        (
            "Write each list item as a mini-review, not a bullet. "
            "Each item needs: what it does, one concrete strength, one concrete weakness, "
            "and who it's best for. "
            "The FAQ section must have at least 4 real questions — write the questions "
            "a beginner would actually search, not the ones an expert would ask. "
            "This format performs well for search — optimise the headings for question-based queries."
        ),
    ),
    (
        "troubleshooting",
        [
            "## The error and why it's confusing",
            "## What's actually causing it (the real reason, not the surface symptom)",
            "## Fix 1 — the most common cause",
            "## Fix 2 — the less obvious cause",
            "## Fix 3 — the environment-specific cause",
            "## How to verify the fix worked",
            "## How to prevent this from happening again",
            "## Related errors you might hit next",
            "## When none of these work: escalation path",
        ],
        (
            "Write this as a diagnostic guide, not a tutorial. "
            "Start each 'Fix' section by describing the symptom pattern that indicates this cause — "
            "so the reader can self-triage before reading the solution. "
            "Include the exact error message text where relevant (AdSense loves exact-match search content). "
            "The 'related errors' section is important for internal linking — name them specifically."
        ),
    ),
]


def _pick_structure(topic: str) -> tuple:
    idx = int(hashlib.md5(topic.encode()).hexdigest(),
              16) % len(_STRUCTURE_SETS)
    return _STRUCTURE_SETS[idx]


# ─────────────────────────────────────────────────────────────────
# Author persona contexts
# ─────────────────────────────────────────────────────────────────

_AUTHOR_CONTEXTS = [
    (
        "You are Kubai Kevin, a software engineer in Nairobi with 10+ years building "
        "production Python and Node.js backends in fintech. You write from direct experience — "
        "name specific AWS services you've used, recall a real incident, mention a library version "
        "that bit you. Never claim to work at a company you didn't. Write like you're explaining "
        "to a smart colleague at a Nairobi tech meetup."
    ),
    (
        "You are Kubai Kevin, a developer who spends a lot of time reading GitHub issues, "
        "production postmortems, and Hacker News comment threads. You're opinionated and specific. "
        "You've seen hype cycles come and go. Write with earned skepticism — praise what deserves "
        "praise, call out what's overrated. Your audience respects directness."
    ),
    (
        "You are Kubai Kevin, a self-taught developer who learned by breaking things in production. "
        "You write the guide you wished existed when you were learning this topic. "
        "Include at least one thing that took you longer than it should have to figure out. "
        "Acknowledge when something is genuinely hard, not just 'initially confusing'."
    ),
    (
        "You are Kubai Kevin, a developer who reviews a lot of code and sees the same mistakes repeatedly. "
        "This post is your attempt to address the root cause, not just the symptom. "
        "Be empathetic — most mistakes come from following outdated tutorials, not incompetence. "
        "Name the outdated pattern before showing the better one."
    ),
    (
        "You are Kubai Kevin, a remote engineer who has worked with distributed teams across "
        "Lagos, Berlin, Singapore, and San Francisco — sometimes on the same project. "
        "You've learned that 'best practices' are often region-specific: what works smoothly "
        "on a US-East server at 50ms latency hits differently on a shared VPS in West Africa. "
        "Write with that gap in mind. Name the constraint before naming the solution."
    ),
    (
        "You are Kubai Kevin, a contractor who has billed clients in Europe, the US, and the Gulf. "
        "Your readers aren't all in Silicon Valley — some are bootstrapping on $200/month DigitalOcean "
        "droplets, others are at Series B startups with AWS enterprise agreements. "
        "When you recommend a tool, say which budget tier it actually makes sense for."
    ),
    (
        "You are Kubai Kevin, an engineer who has shipped products used heavily in Nigeria, Ghana, "
        "and East Africa. You've debugged issues that only showed up on mobile data connections, "
        "optimised for users on 3G, and built payment integrations for M-Pesa, Flutterwave, and Paystack. "
        "When you write about performance or reliability, 'good enough for Chrome on fibre' is not "
        "the bar — mobile-first, intermittent-connection-tolerant is. Name that constraint explicitly."
    ),
    (
        "You are Kubai Kevin, a developer who has worked on government and NGO tech projects "
        "across sub-Saharan Africa. You've shipped things with real constraints: no credit card "
        "for AWS, users on feature phones, unreliable power during deployment windows. "
        "Your posts reflect that not every team has a devops engineer or a $10k/month cloud budget. "
        "Practical alternatives to expensive tooling matter here."
    ),
    (
        "You are Kubai Kevin, a backend engineer who has spent time working with European clients "
        "where GDPR compliance, data residency, and audit trails are non-negotiable. "
        "You've had to explain to product managers why you can't just use a US-hosted SaaS for PII. "
        "When a topic touches data handling, storage, or third-party integrations, you factor "
        "compliance in from the start — not as an afterthought bolted on before launch."
    ),
    (
        "You are Kubai Kevin, a developer who has worked with startups in Southeast Asia — "
        "Indonesia, Vietnam, the Philippines — where the goal is often 'scale to millions of users "
        "before Series A'. You've seen architectures that handle enormous traffic on surprisingly "
        "lean infrastructure. When you talk about cost optimisation, you mean it: you've actually "
        "cut bills, not just theorised about it. Quote real numbers when you have them."
    ),
    (
        "You are Kubai Kevin, a freelance engineer who has built products for clients in Brazil, "
        "Colombia, and Mexico. You know what it's like to work in a timezone that doesn't overlap "
        "neatly with your client's, to deal with payment processors that don't support your region, "
        "and to build resilient systems when managed Kubernetes isn't in the budget. "
        "Your writing is grounded in that context — real tradeoffs, not ideal-world advice."
    ),
    (
        "You are Kubai Kevin, a developer who has contributed to open source projects and "
        "maintains a few small libraries used by engineers in several countries. "
        "You write for a global audience that includes beginners in Accra reading on a phone, "
        "senior engineers in London skimming for one specific insight, and students in India "
        "following along to build their first production-grade project. "
        "Write clearly enough for the beginner, specifically enough to be useful to the senior."
    ),
    (
        "You are Kubai Kevin, a developer with opinions forged by a decade of watching hype cycles "
        "burn through the industry. You've seen blockchain, serverless, microservices, and now AI "
        "all get oversold and then quietly normalised. "
        "Your writing cuts through the marketing language: what does this actually do, "
        "what does it actually cost, and what breaks first under real load? "
        "Your audience is global — developers in Lagos, London, Manila, and Montreal — "
        "and they all appreciate the same thing: honesty about tradeoffs."
    ),
    (
        "You are Kubai Kevin, writing specifically for developers who are 1–4 years into their careers "
        "and trying to cross the gap between 'it works on my machine' and 'it works in production'. "
        "Your audience is global — bootcamp grads in Lagos, CS graduates in Bangalore, "
        "self-taught developers in São Paulo. The knowledge gap is the same everywhere: "
        "tutorials show the happy path, production doesn't have one. "
        "Write the guide that closes that gap."
    ),
    (
        "You are Kubai Kevin, writing for solo founders and indie hackers who are also the "
        "sole engineer on their product. Your reader in Cape Town, Tallinn, or Manila has to "
        "make every architectural decision themselves, maintain it themselves, and explain it "
        "to non-technical co-founders or clients. "
        "Optimise your advice for the person who is both the decision-maker and the implementer. "
        "Flag the decisions that are hard to reverse. Recommend the boring, proven option "
        "unless you have a concrete reason not to."
    ),
    (
        "You are Kubai Kevin, a developer who has done security reviews for fintech and healthtech "
        "products serving users in multiple countries. You've seen auth bugs, insecure direct object "
        "references, and secrets committed to public repos — not in tutorials, but in real codebases. "
        "When you write about any topic that touches auth, data storage, or external APIs, "
        "you fold security in naturally, not as a separate 'security considerations' section "
        "that gets skimmed. Your audience is global; the attack surface is too."
    ),
    (
        "You are Kubai Kevin, a backend engineer who gets unreasonably interested in query plans, "
        "connection pool tuning, and p99 latency. You've profiled Python services, optimised "
        "Postgres indexes, and traced memory leaks in Node.js at 3am. "
        "Your readers are engineers anywhere in the world who are hitting a wall with performance "
        "and need someone to show them where to look first. "
        "Lead with the measurement, not the fix. A developer in Jakarta and one in Dublin "
        "both need to know what to instrument before they can know what to change."
    ),
]


def _build_humanization_note(topic: str) -> str:
    idx = int(hashlib.sha256(topic.encode()).hexdigest(),
              16) % len(_AUTHOR_CONTEXTS)
    return _AUTHOR_CONTEXTS[idx]


# ─────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────

def _build_system_prompt(author_note: str, format_name: str, format_note: str, year_guidance: str) -> str:
    return (
        f"{author_note}\n\n"
        f"{year_guidance}\n\n"
        "VOICE: Write with a specific, personal voice. Use 'I' and 'we' where natural. "
        "You are not a content marketing agency. You are a developer who has actually "
        "hit this problem in production. Write as if explaining to a smart colleague "
        "who has 3 years of experience — skip the basics they already know, but don't "
        "assume they've seen this specific edge case before.\n\n"
        "BANNED PHRASES — never use these, not even once:\n"
        "- 'in today's fast-paced world'\n"
        "- 'it is important to note'\n"
        "- 'crucial aspect'\n"
        "- 'dive into' or 'delve into'\n"
        "- 'In conclusion'\n"
        "- 'leverage' (use 'use' instead)\n"
        "- 'unleash'\n"
        "- 'game-changer'\n"
        "- 'comprehensive guide'\n"
        "- 'this article will explore'\n"
        "- 'seamlessly'\n"
        "- 'revolutionize'\n"
        "- 'cutting-edge'\n"
        "- 'state-of-the-art'\n"
        "- 'harness the power'\n"
        "- 'unlock the potential'\n"
        "- Any phrase that sounds like it belongs in a press release\n\n"
        "ADSENSE REQUIREMENTS — the post will be rejected if it lacks:\n"
        "1. At least ONE first-person sentence about a real mistake or surprise "
        "(e.g. 'I spent three days on this before realising...')\n"
        "2. At least TWO code blocks with language tags\n"
        "3. At least THREE concrete numbers (ms, %, cost, line count, version number)\n"
        "4. At least ONE tool with a specific version number "
        "(e.g. 'Python 3.11', 'Redis 7.2', 'Node 20 LTS')\n"
        "5. A comparison table using markdown table syntax\n"
        "6. A 'Frequently Asked Questions' section with 3-4 real developer questions\n"
        "7. A specific, actionable closing step the reader can do in the next 30 minutes\n\n"
        "CREDIBILITY: Name actual tools. Name actual AWS services. Name specific "
        "error messages you've seen. Be willing to say something is hard or that "
        "you got it wrong at first. Generic advice with no specifics is exactly "
        "what Google's quality raters flag as low-value content.\n\n"
        f"FORMAT: {format_name.upper()} — {format_note}\n\n"
        "IMPORTANT: Respond with ONLY a valid JSON object — no markdown fences, "
        "no preamble, no trailing commentary."
    )


# ─────────────────────────────────────────────────────────────────
# Personal intro injection
# ─────────────────────────────────────────────────────────────────

_PERSONAL_INTROS = [
    (
        "The official documentation for {keyword} is good. What it doesn't cover is what "
        "happens when you're six months into production and the edge cases start appearing. "
        "This is the post that fills that gap."
    ),
    (
        "I spent longer than I should have on this before I understood what was actually happening. "
        "The tutorials all showed the happy path. This post shows what comes after."
    ),
    (
        "A colleague asked me about {keyword} during a code review last week. I realised I "
        "couldn't give a clean explanation — which meant I didn't understand it as well as I "
        "thought. This post is what I put together after properly working through it."
    ),
    (
        "I've seen the same {keyword} mistake in multiple production codebases, including one "
        "I wrote myself three years ago. Here's what it looks like, why it's hard to spot, "
        "and how to fix it."
    ),
    (
        "Most {keyword} guides assume a clean environment and a patient timeline. "
        "Production gives you neither. Here's what I learned building this under real constraints."
    ),
    (
        "The short version: the conventional advice on {keyword} is incomplete. "
        "It works in the simple case, and breaks in a specific way under load. "
        "Here's the fuller picture."
    ),
    (
        "I ran into this {keyword} problem while migrating a service under a hard deadline. "
        "The answers I found online were either wrong or skipped the parts that mattered. "
        "Here's what actually worked."
    ),
    (
        "After reviewing a lot of code that touches {keyword}, I keep seeing the same patterns "
        "that cause problems later. This post addresses the root cause rather than the symptom."
    ),
]


def inject_personal_intro(post, topic: str) -> None:
    topic_lower = topic.lower()
    stop = {"how", "to", "the", "a", "an", "for", "and", "or", "vs",
            "when", "why", "what", "which", "guide", "tutorial", "tips"}
    words = [w for w in re.sub(r'[^\w\s]', '', topic_lower).split()
             if w not in stop and len(w) > 2]
    keyword = " ".join(words[:2]) if words else topic_lower

    idx = int(hashlib.md5(topic.encode()).hexdigest(),
              16) % len(_PERSONAL_INTROS)
    intro_template = _PERSONAL_INTROS[idx]
    intro = intro_template.format(keyword=keyword)

    if intro[:30] not in post.content:
        post.content = f"{intro}\n\n{post.content}"


# ─────────────────────────────────────────────────────────────────
# E-E-A-T signal injection
# ─────────────────────────────────────────────────────────────────

_EEAT_FOOTER_TEMPLATE = """

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** {review_date}
"""


def inject_eeat_signals(post, topic: str) -> None:
    sentinel = "### About this article"
    if sentinel in post.content:
        return
    review_date = datetime.now().strftime("%B %d, %Y")
    footer = _EEAT_FOOTER_TEMPLATE.format(review_date=review_date)
    post.content = post.content.rstrip() + "\n" + footer


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
        self.github_token = os.getenv("BLOGGITHUB_TOKEN")
        self.cloudflare_token = os.getenv("CLOUDFLARE_API_TOKEN")
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")

        self._log_key_status()

        self.api_key = (
            self.groq_key or self.openrouter_key or self.cerebras_key
            or self.mistral_key or self.nvidia_key or self.gemini_key
            or self.github_token or self.cloudflare_token
        )

        self.monetization = MonetizationManager(config)
        self.hashtag_manager = HashtagManager(config)

    def _log_key_status(self):
        print("=== API Key Status ===")
        print(
            f"  Groq:           {'configured' if self.groq_key            else 'NOT SET'}")
        print(
            f"  OpenRouter:     {'configured' if self.openrouter_key       else 'NOT SET'}")
        print(
            f"  Cerebras:       {'configured' if self.cerebras_key         else 'NOT SET'}")
        print(
            f"  Mistral:        {'configured' if self.mistral_key          else 'NOT SET'}")
        print(
            f"  NVIDIA NIM:     {'configured' if self.nvidia_key           else 'NOT SET'}")
        print(
            f"  Gemini:         {'configured' if self.gemini_key           else 'NOT SET'}")
        print(
            f"  GitHub Models:  {'configured' if self.github_token         else 'NOT SET'}")
        cf_status = (
            "configured" if (self.cloudflare_token and self.cloudflare_account_id)
            else "PARTIAL — need both CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID" if (self.cloudflare_token or self.cloudflare_account_id)
            else "NOT SET"
        )
        print(f"  Cloudflare AI:  {cf_status}")
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

    def purge_low_quality_posts(self, dry_run: bool = True):
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

    async def _call_api_with_fallback(self, messages: List[Dict], max_tokens: int = 6000) -> str:
        providers = []

        if self.mistral_key:
            providers.append(("Mistral",         self._call_mistral))
        if self.github_token:
            providers.append(("GitHub Models",    self._call_github))
        if self.openrouter_key:
            providers.append(("OpenRouter",       self._call_openrouter))
        if self.groq_key:
            providers.append(("Groq",             self._call_groq))
        if self.cloudflare_token and self.cloudflare_account_id:
            providers.append(("Cloudflare AI",    self._call_cloudflare))
        if self.cerebras_key:
            providers.append(("Cerebras",         self._call_cerebras))
        if self.gemini_key:
            providers.append(("Gemini",           self._call_gemini))
        if self.nvidia_key:
            providers.append(("NVIDIA NIM",       self._call_nvidia))

        if not providers:
            raise Exception(
                "No API keys configured. Set at least one of: GROQ_API_KEY, "
                "OPENROUTER_API_KEY, CEREBRAS_API_KEY, MISTRAL_API_KEY, "
                "NVIDIA_API_KEY, GEMINI_API_KEY, GITHUB_TOKEN, "
                "or CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID."
            )

        _RETRY_DELAYS = [5, 15, 30]
        _MAX_CHAIN_RETRIES = 3

        last_error = None
        for chain_attempt in range(1, _MAX_CHAIN_RETRIES + 1):
            if chain_attempt > 1:
                delay = _RETRY_DELAYS[chain_attempt - 2]
                print(
                    f"All providers failed on attempt {chain_attempt - 1}. "
                    f"Retrying full chain in {delay}s "
                    f"(attempt {chain_attempt}/{_MAX_CHAIN_RETRIES})..."
                )
                await asyncio.sleep(delay)

            for name, caller in providers:
                try:
                    result = await caller(messages, max_tokens)
                    print(f"API: {name} responded successfully "
                          f"(chain attempt {chain_attempt}).")
                    return result
                except Exception as e:
                    last_error = e
                    print(f"{name} error: {e}")
                    if name != providers[-1][0]:
                        print("Falling back to next provider...")

            print(f"Full provider chain exhausted on attempt "
                  f"{chain_attempt}/{_MAX_CHAIN_RETRIES}.")

        raise Exception(
            f"All configured API providers failed after {_MAX_CHAIN_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ─────────────────────────────────────────────────────────────
    # PROVIDERS
    # ─────────────────────────────────────────────────────────────

    async def _call_groq(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.groq_key}",
                   "Content-Type": "application/json"}
        data = {"model": "llama-3.3-70b-versatile", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7}
        waits = [2, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Groq {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Groq connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("Groq timed out.")
        raise Exception("Groq unavailable.")

    async def _call_openrouter(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.get("base_url", "https://kubaik.github.io"),
            "X-Title": self.config.get("site_name", "Tech Blog"),
        }
        data = {
            "model": "openai/gpt-oss-120b:free",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "provider": {"ignore": ["Venice"], "allow_fallbacks": True},
        }
        waits = [2, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            result = await r.json()
                            if "error" in result:
                                raise Exception(
                                    f"OpenRouter error: {result['error']}")
                            return result["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"OpenRouter {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"OpenRouter connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("OpenRouter timed out.")
        raise Exception("OpenRouter unavailable.")

    async def _call_cerebras(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.cerebras_key}",
                   "Content-Type": "application/json"}
        data = {"model": "qwen-3-235b-a22b-instruct-2507",
                "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
        waits = [2, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://api.cerebras.ai/v1/chat/completions", headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Cerebras {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Cerebras connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("Cerebras timed out.")
        raise Exception("Cerebras unavailable.")

    async def _call_mistral(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.mistral_key}",
                   "Content-Type": "application/json"}
        data = {"model": "mistral-small-latest", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7}
        waits = [_MISTRAL_FREE_TIER_DELAY, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Mistral {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Mistral connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("Mistral timed out.")
        raise Exception("Mistral unavailable.")

    async def _call_nvidia(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.nvidia_key}",
                   "Content-Type": "application/json"}
        data = {"model": "meta/llama-3.3-70b-instruct", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7, "stream": False}
        waits = [2, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"NVIDIA NIM {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"NVIDIA NIM connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("NVIDIA NIM timed out.")
        raise Exception("NVIDIA NIM unavailable.")

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
                parts = [
                    ("SYSTEM: " if m.get("role") ==
                     "system" else "USER: ") + m.get("content", "")
                    for m in messages
                ]
                return model.generate_content("\n\n".join(parts) + "\n\nASSISTANT:").text
            return await asyncio.get_event_loop().run_in_executor(None, _sdk_call)
        except ImportError:
            pass

        api_url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
        )
        system_parts = [m["content"]
                        for m in messages if m.get("role") == "system"]
        user_parts = [m["content"]
                      for m in messages if m.get("role") != "system"]
        first_user = (
            ("\n\n".join(system_parts) + "\n\n" if system_parts else "")
            + (user_parts[0] if user_parts else "")
        )
        contents = [{"role": "user", "parts": [{"text": first_user}]}]
        for extra in user_parts[1:]:
            contents.append({"role": "user", "parts": [{"text": extra}]})
        payload = {"contents": contents, "generationConfig": {
            "maxOutputTokens": max_tokens, "temperature": 0.7}}
        waits = [2, 5, 10, 20]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status == 200:
                            result = await r.json()
                            try:
                                return result["candidates"][0]["content"]["parts"][0]["text"]
                            except (KeyError, IndexError) as e:
                                raise Exception(f"Gemini parse error: {e}")
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        raise Exception(f"Gemini {r.status}: {await r.text()}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Gemini connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("Gemini timed out.")
        raise Exception("Gemini unavailable.")

    async def _call_github(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.github_token}",
                   "Content-Type": "application/json"}
        data = {"model": "gpt-4o", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7}
        waits = [2, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(_GITHUB_MODELS_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        body = await r.text()
                        raise Exception(
                            f"GitHub Models {r.status}: {body[:250]}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"GitHub Models connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("GitHub Models timed out.")
        raise Exception("GitHub Models unavailable.")

    async def _call_cloudflare(self, messages: List[Dict], max_tokens: int) -> str:
        RETRYABLE = {503, 429, 500, 502, 504}
        headers = {"Authorization": f"Bearer {self.cloudflare_token}",
                   "Content-Type": "application/json"}
        data = {"model": _CF_MODEL, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.7, "stream": False}
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/v1/chat/completions"
        waits = [2, 5, 10]
        for attempt in range(1, 3):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status == 200:
                            return (await r.json())["choices"][0]["message"]["content"]
                        if r.status in RETRYABLE and attempt < 2:
                            await asyncio.sleep(waits[attempt - 1])
                            continue
                        body = await r.text()
                        raise Exception(
                            f"Cloudflare AI {r.status}: {body[:250]}")
            except aiohttp.ClientConnectionError as e:
                if attempt < 2:
                    await asyncio.sleep(waits[attempt - 1])
                else:
                    raise Exception(f"Cloudflare AI connection failed: {e}")
            except asyncio.TimeoutError:
                raise Exception("Cloudflare AI timed out.")
        raise Exception("Cloudflare AI unavailable.")

    # ─────────────────────────────────────────────────────────────
    # CONTENT GENERATION
    # ─────────────────────────────────────────────────────────────

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> BlogPost:
        if not self.api_key:
            print("No API keys configured. Using local template content.")
            return self._generate_fallback_post(topic)

        SEP = "─" * 60

        attempted_topics: List[str] = []
        current_topic = topic
        current_keywords = keywords

        for attempt_num in range(1, MAX_GENERATION_ATTEMPTS + 1):
            attempted_topics.append(current_topic)
            print(f"\n{SEP}")
            print(
                f"Generation attempt {attempt_num}/{MAX_GENERATION_ATTEMPTS} "
                f"— topic: '{current_topic}'"
            )
            print(SEP)

            existing_titles = _load_existing_titles(self.output_dir)

            try:
                bundle = await self._generate_full_bundle(
                    current_topic, current_keywords, existing_titles
                )
            except Exception as e:
                print(
                    f"Bundle generation failed on attempt {attempt_num}: {e}")
                if attempt_num < MAX_GENERATION_ATTEMPTS:
                    current_topic = self._pick_retry_topic(
                        current_topic, existing_titles, exclude=attempted_topics
                    )
                    current_keywords = None
                    continue
                raise InsufficientContentError(
                    f"All {MAX_GENERATION_ATTEMPTS} generation attempts failed at the "
                    f"bundle stage. Last error: {e}"
                )

            try:
                title = bundle["title"].strip().strip('"')
                _TITLE_FILLER = re.compile(
                    r'^(a\s+|an\s+|the\s+|complete\s+|ultimate\s+|comprehensive\s+|'
                    r'introduction\s+to\s+|guide\s+to\s+|overview\s+of\s+|'
                    r'everything\s+you\s+need\s+to\s+know\s+about\s+)',
                    re.IGNORECASE,
                )
                title = _TITLE_FILLER.sub('', title).strip()

                if len(title) > 55:
                    title = title[:55].rsplit(' ', 1)[0].rstrip(',:;-')

                content = bundle["content"].strip()
                meta_description = bundle["meta_description"].strip()
                seo_keywords = [k.strip()
                                for k in bundle["seo_keywords"] if k.strip()]

                if not meta_description:
                    print(
                        "Warning: meta_description empty from API — deriving from content.")
                    meta_description = _derive_description(content, title)

                _weak_openers = (
                    "this post", "in this article", "a guide to",
                    "learn about", "an overview", "this tutorial",
                    "this article", "we will", "you will learn",
                )
                if any(meta_description.lower().startswith(w) for w in _weak_openers):
                    print("Warning: meta_description has weak opener — re-deriving.")
                    meta_description = _derive_description(content, title)

                if not current_keywords:
                    current_keywords = seo_keywords

                content = _scrub_stale_years(content)
                word_count = _count_words(content)
                print(f"Generated content: {word_count} words")

            except Exception as e:
                print(f"Post-processing error on attempt {attempt_num}: {e}")
                if attempt_num < MAX_GENERATION_ATTEMPTS:
                    current_topic = self._pick_retry_topic(
                        current_topic, existing_titles, exclude=attempted_topics
                    )
                    current_keywords = None
                    continue
                raise InsufficientContentError(
                    f"Post-processing failed on all {MAX_GENERATION_ATTEMPTS} attempts. "
                    f"Last error: {e}"
                )

            if word_count < MIN_WORD_COUNT:
                print(
                    f"Content short ({word_count} words, target ≥ {MIN_WORD_COUNT}). "
                    f"Attempting one expansion pass..."
                )
                try:
                    expanded = await self._expand_content(content, title, current_topic)
                    expanded_count = _count_words(expanded)
                    print(f"After expansion: {expanded_count} words")

                    if expanded_count > word_count:
                        content = _scrub_stale_years(expanded)
                        word_count = _count_words(content)
                    else:
                        print(
                            f"Expansion did not increase word count "
                            f"({word_count} → {expanded_count}). Keeping original."
                        )
                except Exception as e:
                    print(
                        f"Expansion pass failed: {e}. Continuing with original content.")

            if word_count < MIN_ACCEPTABLE_WORDS:
                print(
                    f"\n❌  Attempt {attempt_num}/{MAX_GENERATION_ATTEMPTS} FAILED: "
                    f"content has only {word_count} words "
                    f"(minimum required: {MIN_ACCEPTABLE_WORDS})."
                )
                if attempt_num < MAX_GENERATION_ATTEMPTS:
                    current_topic = self._pick_retry_topic(
                        current_topic, existing_titles, exclude=attempted_topics
                    )
                    current_keywords = None
                    print(f"Switching to new topic: '{current_topic}'")
                    continue
                raise InsufficientContentError(
                    f"Failed to generate adequate content after "
                    f"{MAX_GENERATION_ATTEMPTS} attempts across topics: "
                    + ", ".join(f"'{t}'" for t in attempted_topics)
                    + f". Each attempt produced fewer than {MIN_ACCEPTABLE_WORDS} words. "
                    f"No post has been saved."
                )

            print(
                f"\n✅  Attempt {attempt_num}: content adequate "
                f"({word_count} words ≥ {MIN_ACCEPTABLE_WORDS})."
            )

            existing_titles_now = _load_existing_titles(self.output_dir)
            is_dup, dup_match, dup_score = _is_duplicate_title(
                title, existing_titles_now, threshold=DUPLICATE_TITLE_THRESHOLD
            )
            if is_dup:
                print(
                    f"WARNING: Generated title is a duplicate of an existing post.\n"
                    f"  Generated : '{title}'\n"
                    f"  Existing  : '{dup_match}'\n"
                    f"  Similarity: {dup_score:.0%}\n"
                    f"  Requesting a new title from the LLM..."
                )
                title = await self._regenerate_title(
                    title=title,
                    content=content,
                    topic=current_topic,
                    existing_titles=existing_titles_now,
                )
                print(f"  New title : '{title}'")

            slug = self._create_slug(title)

            # PATCH 11: Compute reading_time_minutes before constructing BlogPost
            reading_time_minutes = max(1, round(word_count / 200))

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
                # PATCH 11
                reading_time_minutes=reading_time_minutes,
            )

            post.affiliate_links = []
            post.monetization_data = self.monetization.generate_ad_slots(
                post.content)

            print("Deriving hashtags from title + keywords (tiered system)...")
            hashtags = _derive_hashtags_from_keywords(
                seo_keywords, topic=current_topic, title=title, max_hashtags=5
            )
            print(f"Hashtags selected: {', '.join(hashtags)}")

            all_tags = _to_single_word_tags(seo_keywords[:5] + hashtags)
            post.tags = all_tags[:15]
            post.seo_keywords = _to_single_word_tags(
                seo_keywords + hashtags)[:15]
            post.twitter_hashtags = " ".join(
                f"#{h.replace(' ', '').replace('-', '')}" for h in hashtags
            )

            bundle_tweet = bundle.get("tweet_text", "").strip()
            if bundle_tweet:
                post_url = (
                    f"{self.config.get('base_url', 'https://kubaik.github.io')}"
                    f"/{post.slug}"
                )

                TCO_LEN = 23
                URL_SEP = 2
                TAG_SEP = 2
                BAIT_SEP = 2
                MAX_TAGS_CHARS = 60
                TWITTER_LIMIT = 280

                from visibility_automator import _get_hashtags_for_post
                hashtag_str = _get_hashtags_for_post(post, max_tags=4)

                if len(hashtag_str) > MAX_TAGS_CHARS:
                    parts = hashtag_str.split()
                    hashtag_str = ""
                    for tag in parts:
                        if len(hashtag_str) + len(tag) + 1 <= MAX_TAGS_CHARS:
                            hashtag_str = (hashtag_str + " " + tag).strip()
                        else:
                            break

                _BAIT_POOL = [
                    "What broke first when you tried this?",
                    "Done this differently? Tell me what worked.",
                    "Which part took you the longest to get right?",
                    "Where do you think most teams still get this wrong?",
                    "What would you add to this?",
                    "Hot take: most people skip the measurement step. Agree?",
                    "What's the tool you wished you'd found earlier?",
                    "Anyone hit a different failure mode? Reply below.",
                ]
                bait_idx = int(hashlib.md5(post.slug.encode()
                                           ).hexdigest(), 16) % len(_BAIT_POOL)
                reply_bait = _BAIT_POOL[bait_idx]

                fixed_cost = URL_SEP + TCO_LEN
                tags_cost = (TAG_SEP + len(hashtag_str)) if hashtag_str else 0
                bait_cost = (BAIT_SEP + len(reply_bait)) if reply_bait else 0
                body_budget = TWITTER_LIMIT - fixed_cost - tags_cost - bait_cost

                if len(bundle_tweet) > body_budget:
                    bundle_tweet = _trim_to_budget(bundle_tweet, body_budget)
                    print(
                        f"Note: tweet body trimmed to {len(bundle_tweet)} chars "
                        f"(budget was {body_budget})."
                    )

                effective = len(bundle_tweet) + fixed_cost + \
                    tags_cost + bait_cost
                if effective > TWITTER_LIMIT:
                    bait_cost = 0
                    reply_bait = ""
                    effective = len(bundle_tweet) + fixed_cost + tags_cost
                    print("Note: reply-bait dropped to fit within 280.")

                if effective > TWITTER_LIMIT:
                    tags_cost = 0
                    hashtag_str = ""
                    effective = len(bundle_tweet) + fixed_cost
                    print("Note: hashtags dropped — body + URL fills budget.")

                parts = [bundle_tweet, post_url]
                if hashtag_str:
                    parts.append(hashtag_str)
                if reply_bait:
                    parts.append(reply_bait)
                # PATCH 12: prewritten_tweet now assigned to model attribute
                post.prewritten_tweet = "\n\n".join(parts)

                print(
                    f"Bundle tweet assembled: {len(post.prewritten_tweet)} raw chars "
                    f"(effective ~{len(bundle_tweet) + fixed_cost + tags_cost + bait_cost} after t.co)\n"
                    f"  Body    : {len(bundle_tweet)} chars (budget {body_budget})\n"
                    f"  Hashtags: {hashtag_str or '(none)'}\n"
                    f"  Bait    : {reply_bait if reply_bait else '(dropped)'}"
                )
            else:
                # PATCH 12: prewritten_tweet assigned to model attribute
                post.prewritten_tweet = ""
                print("Note: no tweet_text in bundle — template fallback will be used.")

            return post

        raise InsufficientContentError(
            f"Exhausted {MAX_GENERATION_ATTEMPTS} generation attempts without "
            f"producing adequate content. No post has been saved."
        )

    # ─────────────────────────────────────────────────────────────
    # BUNDLE GENERATION
    # ─────────────────────────────────────────────────────────────

    async def _generate_full_bundle(
        self,
        topic: str,
        keywords: List[str],
        existing_titles: List[str],
    ) -> dict:
        format_name, headings, format_note = _pick_structure(topic)
        author_note = _build_humanization_note(topic)

        resolved_headings = [h.replace("{topic}", topic) for h in headings]
        heading_block = "\n".join(resolved_headings)

        keyword_text = (
            f"\nKeywords to incorporate naturally: {', '.join(keywords)}"
            if keywords else ""
        )
        existing_hint = (
            " Avoid titles similar to: "
            + ", ".join(f'"{t}"' for t in existing_titles[:8])
            if existing_titles else ""
        )

        title_guidance = {
            "deep_dive":       "Title: MAX 50 chars. Lead with the insight, not the topic. E.g. 'Postgres indexes: the setting nobody checks'.",
            "tutorial":        "Title: MAX 50 chars. Name the outcome + tool only. E.g. 'FastAPI rate limiting in 20 lines'.",
            "opinion":         "Title: MAX 50 chars. State the contrarian take directly. E.g. 'Microservices slowed us down'.",
            "comparison":      "Title: MAX 50 chars. Name both options + the verdict angle. E.g. 'Redis vs Memcached: the benchmark that matters'.",
            "case_study":      "Title: MAX 50 chars. Lead with the result. E.g. 'How we cut latency 60% with one index'.",
            "explainer":       "Title: MAX 50 chars. Name the confusion being resolved. E.g. 'Async Python: when it helps, when it hurts'.",
            "listicle":        "Title: MAX 50 chars. Number + specific promise. E.g. '7 TypeScript traps I keep seeing in code reviews'.",
            "troubleshooting": "Title: MAX 50 chars. Use the exact error symptom. E.g. 'Node.js memory leak: how to find it in 10 min'.",
        }.get(format_name, "Title: MAX 50 chars. Specific, benefit-driven, no filler.")

        year_guidance = (
            "YEAR POLICY: The current year is 2026. All data, statistics, salary "
            "figures, tool versions, and 'as of' references must use 2026 as the "
            "baseline. You may cite research or historical context from earlier years "
            "only when it is explicitly labelled as historical "
            "(e.g. 'a 2024 Stack Overflow survey found...'). "
            "Never present pre-2025 figures as current. "
            "Never write phrases like 'in 2024' or 'as of 2023' without the historical label. "
            "When citing salary ranges, hiring trends, or tool adoption rates, "
            "use 2026 figures or clearly state the year of the source data."
        )

        system_content = _build_system_prompt(
            author_note=author_note,
            format_name=format_name,
            format_note=format_note,
            year_guidance=year_guidance,
        )

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": f"""\
Write a 2500-word {format_name} blog post about: "{topic}"{keyword_text}

{title_guidance}{existing_hint}

GOOGLE ADSENSE CONTENT POLICY — YOUR POST MUST SATISFY ALL OF THESE:
1. Minimum 2000 words of ORIGINAL, substantive content. No filler.
2. At least ONE first-person sentence starting with "I" ("I ran into this when…",
   "I spent two weeks on this…", "I was surprised that…").
3. Named, version-pinned tools and services (e.g. "pytest 7.4", "Node 20 LTS",
   "AWS Lambda with arm64", "Redis 7.2").
4. At least THREE concrete numbers: latency figures, cost savings, benchmark
   results, error rates, salary ranges, or line-of-code counts.
5. A clear point of view — take a side, do not just say "it depends".
6. FORBIDDEN phrases: "In today's fast-paced world", "dive into", "delve into",
   "leverage", "game-changer", "it's important to note", "needless to say",
   "In conclusion", "comprehensive guide", "this article will", "we will explore".
7. The final section must end with ONE specific, actionable next step the reader
   can do today — not "start exploring" or "begin your journey".

Respond with ONLY a JSON object in this exact shape:
{{{{
  "title": "<punchy title: MAX 50 chars. No filler words (Complete/Ultimate/Guide to/Introduction to). Start with a verb, number, or sharp noun. Good: 'Redis caching: what breaks first', 'Cut AWS costs 40%: the real levers', 'TypeScript strict mode traps'. Bad: 'A Complete Guide to Redis Caching'>",
  "content": "<full markdown article — no title heading at top>",
  "meta_description": "<under 155 chars. Must include: (1) a specific number or outcome, (2) the primary keyword, (3) an implied reader benefit. Never start with 'This post', 'In this article', 'A guide to', 'Learn about', 'We will', or 'You will learn'. Good: 'Cut API response time 60% with Redis caching — connection pooling, eviction policies, and the cache stampede mistake most teams make.' Bad: 'A guide to Redis caching for developers.'>",
  "tweet_text": "<X/Twitter hook body — STRICT MAX 180 chars. NO url, NO hashtags (added automatically). Third-person voice only (they/teams/most developers). Complete sentences, no trailing ellipsis. End with an action cue like 'Full breakdown 👇' or 'Here is why 👇'. Good: 'Most teams burn $8k+ on AI tools before measuring ROI.\\n\\nMost of it goes to autocomplete nobody audits.\\n\\nHere is what actually paid off 👇'. Bad: 'I burned $8k...' (first person) or 'Teams overspend on AI... realize...' (truncated).>",
  "seo_keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"]
}}}}

Use EXACTLY these ## headings inside "content" (in order):
{heading_block}

Hard requirements for "content":
- Minimum 2000 words
- At least 2 code examples with language tags (```python, ```javascript, etc.)
- At least 3 concrete numbers (benchmarks, latency ms, percentages, cost figures)
- At least 1 first-person observation: something that surprised you or a mistake you made
- Each section minimum 200 words
- Do NOT include the title as a # heading at the top
- The final section must end with a specific, actionable next step — not a generic "start today"
- "## Frequently Asked Questions" section near the end with 3–4 questions written as
  real search queries (the kind a developer would type into Google).
  Answer each in 3–5 sentences.
- At least one comparison table using markdown table syntax
- AUTHOR VOICE: The introduction must end with one sentence starting with "I" describing
  a real mistake or unexpected result. Example: "I spent three days debugging a connection
  pool issue that turned out to be a single misconfigured timeout — this post is what I
  wished I had found then."
- Closing line of last section: a single, specific action the reader can take in the
  next 30 minutes — name the exact file, command, or metric they should check first.

Requirements for "seo_keywords": 8 items — 2 short-tail (1-2 words), 4 long-tail (3-5 words),
2 question-based (starting with "how", "why", "what", or "when").

Return ONLY the JSON object.""",
            },
        ]

        raw = await self._call_api_with_fallback(messages, max_tokens=6500)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

        data = self._parse_bundle_json(raw)

        for key in ("title", "content"):
            if key not in data:
                raise ValueError(
                    f"Bundle response missing required key: '{key}'")

        if not data.get("meta_description", "").strip():
            print(
                "Note: meta_description missing from API response — deriving from content.")
            data["meta_description"] = _derive_description(
                data.get("content", ""), data.get("title", topic)
            )

        if not data.get("seo_keywords"):
            print("Note: seo_keywords missing — extracting from title/topic.")
            data["seo_keywords"] = [
                topic.lower(),
                f"{topic.lower()} tutorial",
                f"{topic.lower()} guide",
                f"how to use {topic.lower()}",
                f"{topic.lower()} best practices",
                f"{topic.lower()} examples",
                f"what is {topic.lower()}",
                f"{topic.lower()} vs alternatives",
            ]

        if not data.get("tweet_text", "").strip():
            print(
                "Note: tweet_text missing from bundle — template fallback will be used.")

        data["_format"] = format_name
        return data

    # ─────────────────────────────────────────────────────────────
    # Title regeneration
    # ─────────────────────────────────────────────────────────────

    async def _regenerate_title(
        self,
        title: str,
        content: str,
        topic: str,
        existing_titles: List[str],
    ) -> str:
        existing_hint = "\n".join(f'- "{t}"' for t in existing_titles[:20])
        excerpt = " ".join(content.split()[:300])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical blog editor. Your only job right now is to "
                    "produce a single replacement title for a blog post. "
                    "Respond with ONLY the title — no quotes, no explanation, no JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The title '{title}' is too similar to an existing post.\n\n"
                    f"Existing titles to avoid:\n{existing_hint}\n\n"
                    f"Article topic: {topic}\n"
                    f"Article excerpt (first 300 words): {excerpt}\n\n"
                    "Write ONE new title that:\n"
                    "- Is under 55 characters\n"
                    "- Covers the same subject from a different angle\n"
                    "- Is meaningfully distinct from every title in the list above\n"
                    "- Uses no filler words (Complete, Ultimate, Guide to, Introduction to)\n"
                    "- Starts with a verb, number, or sharp noun\n\n"
                    "Respond with ONLY the title text."
                ),
            },
        ]
        try:
            raw = await self._call_api_with_fallback(messages, max_tokens=60)
            new_title = raw.strip().strip('"').strip("'")
            is_still_dup, match, score = _is_duplicate_title(
                new_title, existing_titles, threshold=DUPLICATE_TITLE_THRESHOLD
            )
            if is_still_dup:
                print(
                    f"  Regenerated title '{new_title}' is still similar to '{match}' "
                    f"({score:.0%}). Keeping regenerated version anyway "
                    f"(manual review recommended)."
                )
            return new_title if new_title else title
        except Exception as e:
            print(
                f"  Title regeneration failed ({e}). Keeping original title.")
            return title

    # ─────────────────────────────────────────────────────────────
    # JSON REPAIR / PARSE
    # ─────────────────────────────────────────────────────────────

    def _parse_bundle_json(self, raw: str) -> dict:

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
            rep = text
            if in_str:
                rep += '"'
            for _ in range(max(0, rep.count('[') - rep.count(']'))):
                rep += ']'
            for _ in range(max(0, rep.count('{') - rep.count('}'))):
                rep += '}'
            return rep

        def _fix_unquoted_content(text: str) -> str:
            unquoted_pattern = re.compile(
                r'("content"\s*:\s*)([^"\s\{][^}]*?)(\s*,\s*"(?:meta_description|tweet_text|seo_keywords)"|\s*\})',
                re.DOTALL,
            )
            m = unquoted_pattern.search(text)
            if not m:
                return text
            prefix = m.group(1)
            content = m.group(2)
            suffix = m.group(3)
            content = content.replace('\\n', '\n')
            encoded = json.dumps(content)
            return text[:m.start()] + prefix + encoded + suffix + text[m.end():]

        def _partial(text):
            data = {}
            m = re.search(
                r'"title"\s*:\s*"(.*?)(?:"\s*,|\"\s*\})', text, re.DOTALL)
            if m:
                data['title'] = m.group(1).replace('\\"', '"').strip()

            m = re.search(
                r'"content"\s*:\s*"(.*?)(?:"\s*,\s*"(?:meta_description|seo_keywords|tweet_text)|"\s*\})',
                text, re.DOTALL,
            )
            if not m:
                m = re.search(r'"content"\s*:\s*"(.*)', text, re.DOTALL)
            if m:
                data['content'] = (
                    m.group(1)
                    .replace('\\n', '\n')
                    .replace('\\"', '"')
                    .replace('\\t', '\t')
                )
            else:
                m2 = re.search(
                    r'"content"\s*:\s*([^"\{][^}]*?)(?=,\s*"(?:meta_description|tweet_text|seo_keywords)"|\s*\})',
                    text, re.DOTALL,
                )
                if m2:
                    data['content'] = m2.group(1).strip().rstrip(',').strip()

            m = re.search(
                r'"meta_description"\s*:\s*"(.*?)(?:"\s*,\s*"|\"\s*\})', text, re.DOTALL)
            if m:
                data['meta_description'] = m.group(
                    1).replace('\\n', ' ').strip()

            m = re.search(
                r'"tweet_text"\s*:\s*"(.*?)(?:"\s*,\s*"|\"\s*\})', text, re.DOTALL)
            if m:
                data['tweet_text'] = (
                    m.group(1)
                    .replace('\\n', '\n')
                    .replace('\\"', '"')
                    .strip()
                )

            m = re.search(r'"seo_keywords"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if m:
                data['seo_keywords'] = [
                    k.strip().strip('"')
                    for k in m.group(1).split(',')
                    if k.strip().strip('"')
                ]
            return data

        for attempt in [
            lambda t: json.loads(t),
            lambda t: json.loads(_sanitize(t)),
            lambda t: json.loads(_sanitize(_fix_unquoted_content(t))),
            lambda t: json.loads(_sanitize(
                re.search(r'\{.*\}', t, re.DOTALL).group()
            )) if re.search(r'\{.*\}', t, re.DOTALL) else (_ for _ in ()).throw(ValueError()),
            lambda t: json.loads(_sanitize(_repair(t))),
            lambda t: json.loads(_sanitize(_repair(_fix_unquoted_content(t)))),
        ]:
            try:
                return attempt(raw)
            except Exception:
                pass

        print("Warning: JSON unrecoverable — extracting fields individually.")
        data = _partial(raw)
        if 'content' in data:
            data.setdefault('title', '')
            data.setdefault('meta_description', '')
            data.setdefault('tweet_text', '')
            data.setdefault('seo_keywords', [])
            return data

        raise ValueError(
            f"Model did not return valid JSON.\nRaw (first 400):\n{raw[:400]}"
        )

    # ─────────────────────────────────────────────────────────────
    # EXPANSION
    # ─────────────────────────────────────────────────────────────

    async def _expand_content(self, existing_content: str, title: str, topic: str) -> str:
        author_note = _build_humanization_note(topic)
        messages = [
            {
                "role": "system",
                "content": (
                    f"{author_note}\n\n"
                    "You are expanding an existing blog post. Match the voice and style exactly. "
                    "No generic padding — every sentence must add specific value. "
                    "The current year is 2026. All data, statistics, and tool versions "
                    "must use 2026 as the reference year."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The following blog post about '{topic}' needs more depth. "
                    "Add 3 additional sections at the end (each 300+ words):\n"
                    "1. Advanced edge cases you personally encountered — name them specifically\n"
                    "2. Integration with 2–3 real tools (name versions), with a working code snippet\n"
                    "3. A before/after comparison with actual numbers (latency, cost, lines of code, etc.)\n\n"
                    f"Existing content:\n{existing_content}\n\n"
                    "Return the COMPLETE article — every word of the original content first, "
                    "then the 3 new sections appended at the end. "
                    "Do not summarise, truncate, or paraphrase the original. "
                    "Do not repeat the title. Keep the same author voice throughout. "
                    "The response must be longer than the input."
                ),
            },
        ]
        return await self._call_api_with_fallback(messages, max_tokens=6500)

    # ─────────────────────────────────────────────────────────────
    # LOCAL FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _generate_fallback_post(self, topic: str):
        raise InsufficientContentError(
            f"All API providers failed for topic: '{topic}'. "
            "No fallback post saved — a generic boilerplate post would harm "
            "AdSense approval (Replicated Content violation). "
            "Check your API keys and retry."
        )

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _create_slug(self, title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        return slug.strip('-')[:60]

    def _pick_retry_topic(
        self,
        failed_topic: str,
        existing_titles: List[str],
        exclude: List[str] = None,
    ) -> str:
        import random as _random

        exclude = exclude or []

        history_file = ".used_topics.json"
        used = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    used = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                used = []

        all_topics = self.config.get("content_topics", [])

        candidates = [
            t for t in all_topics
            if t != failed_topic and t not in exclude and t not in used
        ]

        if not candidates:
            candidates = [
                t for t in all_topics
                if t != failed_topic and t not in exclude
            ]

        if not candidates:
            candidates = [t for t in all_topics if t != failed_topic]

        if not candidates:
            return failed_topic

        if existing_titles:
            safe = []
            for candidate in candidates:
                is_dup, _, _ = _is_duplicate_title(
                    candidate, existing_titles, threshold=DUPLICATE_TITLE_THRESHOLD
                )
                if not is_dup:
                    safe.append(candidate)
            if safe:
                candidates = safe

        chosen = _random.choice(candidates)

        used.append(chosen)
        with open(history_file, "w") as f:
            json.dump(used, f, indent=2)

        print(f"Retry topic selected and marked used: {chosen}")
        return chosen

    def save_post(self, post):
        word_count = len(post.content.split())
        # PATCH 11: derive reading_time from content if not set on the model
        reading_time = getattr(post, 'reading_time_minutes', None) or max(
            1, round(word_count / 200))

        if word_count < 1500:
            raise ValueError(
                f"Refusing to save '{post.title}': only {word_count} words. "
                "Minimum is 1500. This post would harm AdSense approval."
            )

        existing_json = self.output_dir / post.slug / "post.json"
        if existing_json.exists():
            try:
                import json as _json
                with open(existing_json, "r", encoding="utf-8") as _f:
                    _existing = _json.load(_f)
                if _existing.get("title", "").strip() != post.title.strip():
                    raise ValueError(
                        f"Slug collision: '{post.slug}' already belongs to "
                        f"'{_existing['title']}'. Refusing to overwrite with "
                        f"'{post.title}'. Change the new post's title or delete "
                        f"the existing post first."
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        if not getattr(post, 'meta_description', '').strip():
            post.meta_description = _derive_description(
                post.content, post.title)
            print("  meta_description was empty — derived from content.")

        # Sync reading_time_minutes onto the model before to_dict()
        post.reading_time_minutes = reading_time

        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        post_data = post.to_dict()
        post_data['word_count'] = word_count
        post_data['reading_time_minutes'] = reading_time
        post_data['has_code'] = '```' in post.content
        post_data['has_table'] = '|' in post.content

        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=2, ensure_ascii=False)

        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")

        print(
            f"Saved: {post.title} ({post.slug}) — "
            f"{word_count} words / ~{reading_time} min read"
        )
        if post.affiliate_links:
            print(f"  - {len(post.affiliate_links)} affiliate links")
        print(
            f"  - has_code={post_data['has_code']} | has_table={post_data['has_table']}")


# ─────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────

class InsufficientContentError(Exception):
    """Raised when generate_blog_post exhausts all retry attempts."""


# ─────────────────────────────────────────────────────────────────
# Stale year scrubber
# ─────────────────────────────────────────────────────────────────

_STALE_YEARS = {"2020", "2021", "2022", "2023", "2024", "2025"}

_HISTORICAL_MARKERS = re.compile(
    r'\b(survey|report|study|research|data|found|published|showed|according|released|'
    r'as of|back in|historically|in a|the \d{4}|a \d{4})\b',
    re.IGNORECASE,
)


def _trim_to_budget(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text

    window = text[:budget]

    for punct in ('.', '!', '?'):
        pos = window.rfind(punct)
        if pos >= budget // 2:
            candidate = text[:pos + 1].rstrip()
            if len(candidate) <= budget:
                return candidate

    for sep in ('—', ';'):
        pos = window.rfind(sep)
        if pos >= budget // 2:
            candidate = text[:pos].rstrip().rstrip(',;')
            if candidate:
                return candidate + '…'

    pos = window.rfind(',')
    if pos >= budget // 2:
        candidate = text[:pos].rstrip()
        if candidate:
            return candidate + '…'

    pos = window.rfind(' ')
    if pos > 0:
        candidate = text[:pos].rstrip('.,;: ')
        return candidate + '…'

    return window.rstrip() + '…'


def _scrub_stale_years(text: str) -> str:
    code_blocks: list = []

    def _mask_code(m):
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    text = re.sub(r'```[\s\S]*?```', _mask_code, text)
    text = re.sub(r'`[^`\n]+`', _mask_code, text)

    iso_dates: list = []

    def _mask_iso(m):
        iso_dates.append(m.group(0))
        return f"\x00ISO{len(iso_dates) - 1}\x00"

    text = re.sub(r'\b(202[0-5])-\d{2}-\d{2}\b', _mask_iso, text)

    def _replace_year(m):
        year = m.group(0)
        if year not in _STALE_YEARS:
            return year
        start = max(0, m.start() - 80)
        preceding = text[start:m.start()]
        if _HISTORICAL_MARKERS.search(preceding):
            return year
        return "2026"

    text = re.sub(r'\b202[0-5]\b', _replace_year, text)

    for i, block in enumerate(iso_dates):
        text = text.replace(f"\x00ISO{i}\x00", block)
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODE{i}\x00", block)

    return text


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
        "hook_style": "auto",
        "social_accounts": {
            "twitter":  "@KubaiKevin",
            "linkedin": "your-linkedin-page",
            "facebook": "your-facebook-page",
        },
        "content_topics": [
            "Copilot vs Cursor: which actually speeds up Python backend work",
            "How I use Claude to review my own code (and where it fails)",
            "RAG pipelines in production: what the tutorials skip",
            "Fine-tuning vs prompt engineering: when each one wins",
            "AI tools that replaced paid software in my workflow",
            "Why most AI-generated code breaks in edge cases",
            "Building a local LLM setup that actually runs on a laptop",
            "How vector databases work (and when you don't need one)",
            "Getting a remote tech job: what actually worked",
            "Tech salaries in 2026: what the data shows",
            "How I went from junior to senior without a CS degree",
            "Freelance developer rates: a realistic breakdown",
            "Why senior developers leave big tech (it's not always the money)",
            "Negotiating a remote salary when you're in a lower-cost country",
            "PostgreSQL vs MySQL in 2026: the decision I keep making",
            "Redis vs Memcached: a benchmark that actually matters",
            "When to use a message queue (and when it's overkill)",
            "GitHub Actions vs CircleCI: real cost comparison at 50k builds/month",
            "How I cut AWS costs by 40% without changing the architecture",
            "API rate limiting patterns that don't break your clients",
            "Database connection pooling: the setting everyone gets wrong",
            "Python async: where it helps and where it makes things worse",
            "TypeScript mistakes that slip through even with strict mode",
            "FastAPI vs Django REST: after building both in production",
            "Node.js memory leaks: how to find and fix them",
            "Python packaging in 2026: what to actually use",
            "Burnout as a freelance developer: what helped me recover",
            "How I manage client work and side projects without losing weekends",
            "The tools that save me the most time as a solo developer",
            "Claude Code vs Cursor: a real cost breakdown after 3 months",
            "How I delegate tasks to AI agents without losing control of my codebase",
            "Multi-agent systems in production: what nobody tells you upfront",
            "Agentic AI for solo developers: where it saves hours and where it wastes them",
            "Why 84% of developers use AI coding tools but only 29% trust the output",
            "Repository intelligence: how AI is learning your entire codebase context",
            "The $100/month AI coding budget question: is it worth it for indie developers",
            "How to review AI-generated code without spending more time than writing it yourself",
            "Vibe coding is fun for prototypes — here's why I stopped using it in production",
            "Building your first agentic workflow with Claude Code: a practical walkthrough",
            "Platform engineering explained: why your team needs a paved road",
            "DevSecOps in practice: shifting security left without slowing down deploys",
            "How to build an internal developer platform on a startup budget",
            "AI-generated code and security: the new vulnerabilities teams are missing",
            "Supply chain attacks in 2026: how to protect your Python and npm dependencies",
            "Zero-trust architecture for small teams: what actually makes sense to implement",
            "AI is a force multiplier: why it makes senior devs more powerful and juniors more exposed",
            "How AI tools are changing what 'senior developer' actually means in 2026",
            "The AI productivity tax: when your tools create more work than they save",
            "Measuring the real impact of AI on your team's velocity (not just vibes)",
            "AI skills that actually affect your salary in 2026: a data-backed breakdown",
            "What happens to junior developers in a world of AI-assisted code generation",
            "How developers in Nairobi and Lagos are landing $4k/month remote roles",
            "Andela, Toptal, and Arc: which platform actually works for African developers in 2026",
            "Building a portfolio that gets you hired remotely from Africa",
            "M-Pesa, Flutterwave, Paystack: integrating African payment APIs without the headaches",
            "The timezone advantage: why European startups are hiring East African engineers",
            "From local salary to global rate: the career moves that made the difference for me",
            "How to pass technical interviews for remote roles when you're self-taught",
            "Infrastructure constraints in Africa that changed how I write backend code",
            "Sustainable software engineering: cutting cloud carbon without cutting performance",
            "Carbon-aware deployment: scheduling workloads when the grid is cleanest",
            "How to cut your AWS bill by 40% using Graviton and spot instances",
            "The real cost of over-engineering: when simplicity beats the fancy architecture",
            "WebSockets vs Server-Sent Events vs long polling: which one for your use case",
            "Edge functions in 2026: when Cloudflare Workers and Vercel Edge actually make sense",
            "Building real-time features without a WebSocket server: practical alternatives",
            "5G and mobile-first backends: what changes when your users are always on cellular",
            "When to build with no-code vs write the code yourself: a decision framework",
            "Non-traditional developers shipping real products: what the AI coding wave made possible",
            "How domain experts (doctors, lawyers, scientists) are building production software in 2026",
            "MCP servers explained: what they are and why every developer should understand them",
            "Designing systems for AI-first applications: the patterns that actually hold up",
            "Event sourcing in 2026: when it's worth the complexity and when it isn't",
            "How to design a multi-tenant SaaS database without painting yourself into a corner",
            "Monolith vs microservices in 2026: the pendulum is swinging back and here's why",
            "Observability for AI pipelines: what to instrument when your system includes an LLM",
        ],
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("Created sample config.yaml")
    print(
        "\nAdd GitHub secrets: GROQ_API_KEY, OPENROUTER_API_KEY, "
        "CEREBRAS_API_KEY, MISTRAL_API_KEY, NVIDIA_API_KEY, GEMINI_API_KEY, "
        "GITHUB_TOKEN, CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID"
    )


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
                "Done! API chain: Mistral → GitHub Models → OpenRouter → Groq → "
                "Cloudflare AI → Cerebras → Gemini → NVIDIA NIM → local template"
            )

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

            except InsufficientContentError as e:
                print("\n" + "═" * 68)
                print("🛑  GENERATION ABORTED — NO POST SAVED")
                print("═" * 68)
                print(f"  Reason : {e}")
                print(
                    f"  Action : Increase content_topics diversity in config.yaml,\n"
                    f"           check API provider quotas, or raise MAX_GENERATION_ATTEMPTS\n"
                    f"           (currently {MAX_GENERATION_ATTEMPTS}) in blog_system.py."
                )
                print("═" * 68 + "\n")
                sys.exit(1)

            except Exception as e:
                print(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            quality_warnings, hard_failures = _validate_content_quality(
                blog_post.content, blog_post.title
            )

            guard = SimilarityGuard(docs_dir=blog_system.output_dir)
            result = guard.check(blog_post)
            if result.is_blocked:
                print(f"🛑 SIMILARITY BLOCK: {result.reason}")
                sys.exit(1)
            if result.warnings:
                for w in result.warnings:
                    print(f"  ⚠️  Similarity warning: {w}")

            if hard_failures:
                print(f"\n🛑  HARD QUALITY FAILURES — post will NOT be saved:")
                for failure in hard_failures:
                    print(f"   ✗ {failure}")
                print()
                print("   This post has been aborted. No file was written.")
                print("   Fix the issues above or regenerate with a new topic.")
                sys.exit(1)

            if quality_warnings:
                print(
                    f"\n⚠️  Content quality warnings ({len(quality_warnings)}):")
                for w in quality_warnings:
                    print(f"   • {w}")
                print()
            else:
                print("✅  Content quality check passed (0 warnings).")

            inject_personal_intro(blog_post, topic)
            inject_eeat_signals(blog_post, topic)

            # ── ADSENSE FIX B: Internal link injection ────────────────────────
            # AdSense guide: "Orphan pages with no internal links signal low quality."
            try:
                posts_index = build_posts_index(blog_system.output_dir)
                base_path = config.get("base_path", "")
                inject_internal_links(
                    blog_post, posts_index, base_path=base_path)
            except Exception as e:
                print(f"  ⚠️  Internal link injection failed (non-fatal): {e}")
            # ─────────────────────────────────────────────────────────────────

            removed = validate_post_links(blog_post, blog_system.output_dir)
            if removed:
                print(f"  🔗 Link validator removed {len(removed)} unresolvable link(s): "
                      f"{', '.join(removed)}")
            blog_system.save_post(blog_post)

            generator = StaticSiteGenerator(blog_system)
            generator.generate_site()

            print(f"\nPost '{blog_post.title}' generated successfully!")
            print(f"Twitter hashtags: {blog_post.twitter_hashtags}")

            visibility = VisibilityAutomator(config)
            prewritten = getattr(blog_post, "prewritten_tweet", "").strip()

            if prewritten:
                final_tweet_text = prewritten
                tweet_source = "bundle (LLM-generated during content creation)"
            else:
                preview = visibility.compose_tweet_preview(blog_post)
                final_tweet_text = preview["tweet_text"]
                tweet_source = f"template fallback (hook_style={preview['hook_style']})"

            SEP = "─" * 68
            print(SEP)
            print("📝  TWEET PREVIEW (always logged)")
            print(SEP)
            print(f"  Post title    : {blog_post.title}")
            print(f"  Slug          : {blog_post.slug}")
            print(f"  Source        : {tweet_source}")
            print(f"  Char count    : {len(final_tweet_text)} / 280")
            print(SEP)
            print("  Full tweet text:")
            print(SEP)
            for line in final_tweet_text.splitlines():
                print(f" {line}")
            print(SEP + "\n")

            if not _twitter_posting_enabled():
                print(
                    "⏭️  Twitter posting SKIPPED (ENABLE_TWITTER_POSTING != true).")
                print("  ↑ Tweet above is what would have been posted.\n")
            else:
                print("Posting tweet...")
                post_result = visibility.post_prewritten_tweet(
                    blog_post, final_tweet_text)

                if post_result["success"]:
                    print(SEP)
                    print("✅  X / TWITTER — POST COMPLETE")
                    print(SEP)
                    print(f"  URL           : {post_result['url']}")
                    print(f"  Tweet ID      : {post_result['tweet_id']}")
                    print(
                        f"  Char count    : {post_result['char_count']} / 280")
                    print(SEP + "\n")
                else:
                    print("❌  X / TWITTER — POST FAILED (no retry)")
                    print(f"  Error         : {post_result.get('error')}")

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

        elif mode == "audit":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            blog_system.purge_low_quality_posts(dry_run=True)

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

        elif mode == 'audit-links':
            from adsense_fixes.link_validator import audit_all_internal_links
            report = audit_all_internal_links(Path('./docs'))
            print(report)

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
                                f"    {fname}: {'Yes' if (item / fname).exists() else 'No'}")
                        if (item / "post.json").exists():
                            try:
                                with open(item / "post.json") as f:
                                    data = json.load(f)
                                wc = _count_words(data.get('content', ''))
                                is_fb = data.get('monetization_data', {}).get(
                                    'used_fallback', False)
                                has_tweet = bool(
                                    data.get('prewritten_tweet', ''))
                                print(
                                    f"    Title: {data.get('title', 'Unknown')} | "
                                    f"Words: {wc} {'✓' if wc >= MIN_WORD_COUNT else '⚠'} "
                                    f"{'[FALLBACK]' if is_fb else ''} "
                                    f"{'[HAS TWEET]' if has_tweet else '[NO TWEET]'}"
                                )
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

        elif mode == "fix-descriptions":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            docs_dir = blog_system.output_dir
            fixed = 0
            for post_dir in docs_dir.iterdir():
                if not post_dir.is_dir() or post_dir.name == "static":
                    continue
                post_json = post_dir / "post.json"
                if not post_json.exists():
                    continue
                try:
                    with open(post_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    desc = data.get("meta_description", "").strip()
                    _weak_openers = (
                        "this post", "in this article", "a guide to",
                        "learn about", "an overview", "this tutorial",
                        "this article", "we will", "you will learn",
                    )
                    needs_fix = (
                        not desc
                        or any(desc.lower().startswith(w) for w in _weak_openers)
                    )
                    if needs_fix:
                        derived = _derive_description(
                            data.get("content", ""), data.get("title", ""))
                        data["meta_description"] = derived
                        with open(post_json, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        reason = "empty" if not desc else "weak opener"
                        print(
                            f"Fixed ({reason}): {post_dir.name} → {derived[:80]}…")
                        fixed += 1
                except Exception as e:
                    print(f"Error fixing {post_dir.name}: {e}")
            print(
                f"\nFixed {fixed} posts. Run 'python blog_system.py build' to regenerate HTML.")

        else:
            print(
                "Usage: python blog_system.py [init|auto|build|cleanup|audit|purge|"
                "debug|social|test-twitter|dedup|fix-descriptions|velocity]"
            )

    else:
        print("AI Blog System — Usage: python blog_system.py [command]")
        print("Commands: init | auto | build | cleanup | audit | purge | debug | social | test-twitter | dedup | fix-descriptions | velocity")
