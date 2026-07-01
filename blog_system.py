import os
import json
import random
import re
import yaml
import asyncio
import aiohttp
import requests
import hashlib
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

from blog_post import BlogPost
from monetization_manager import MonetizationManager
from seo_optimizer import SEOOptimizer
from visibility_automator import VisibilityAutomator
from static_site_generator import StaticSiteGenerator
from hashtag_manager import HashtagManager, add_hashtags_to_post


from adsense_fixes.internal_linker import build_posts_index, inject_internal_links

from velocity_controller import VelocityController
from adsense_fixes.link_validator import validate_post_links
from adsense_fixes.similarity_guard import SimilarityGuard
from adsense_fixes.image_optimizer import inject_alt_text, generate_og_card
from adsense_fixes.canonical_guard import validate_canonical, audit_duplicate_slugs
from adsense_fixes.schema_validator import extract_and_build_faq_schema
from adsense_fixes.content_freshness import inject_freshness_footer, get_publishing_schedule_status
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

# PATCH-2: tightened from 24 → 20
_HASHTAG_MAX_SOURCE_WORDS = 3
_HASHTAG_MAX_CHARS = 20

# Stale content refresh threshold (days)
STALE_THRESHOLD_DAYS = 90

# PATCH-2: question starters to filter from hashtag generation
_QUESTION_STARTERS = {
    "how", "what", "why", "when", "where", "which", "who", "is", "are",
    "does", "do", "can", "should", "will", "would", "could",
}


def _to_single_word_tags(tags: List[str]) -> List[str]:
    # PATCH-2: filters question-starter tags, tighter char cap, fallback to first word
    result = []
    seen: set = set()

    for tag in tags:
        tag = tag.lstrip('#').strip()
        if not tag:
            continue

        words = [w for w in re.split(r'[\s\-_/]+', tag) if w]

        # PATCH-2 FIX: skip question-phrase tags entirely — they make terrible
        # hashtags and signal low-quality content to automated reviewers.
        if words and words[0].lower() in _QUESTION_STARTERS:
            continue

        if len(words) > _HASHTAG_MAX_SOURCE_WORDS:
            continue

        camel = ''.join(w.capitalize() for w in words if w)

        # If CamelCase is still too long after word-count cap, try just the
        # first meaningful word so we always emit something usable.
        if len(camel) > _HASHTAG_MAX_CHARS:
            camel = words[0].capitalize() if words else ''

        if not camel or len(camel) > _HASHTAG_MAX_CHARS or len(camel) < 2:
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
# Content quality validation — PATCH-3 applied
# ─────────────────────────────────────────────────────────────────

def _validate_content_quality(content: str, title: str):
    # PATCH-3: removed over-aggressive title-overlap hard failure;
    # replaced with a high-threshold (0.95) warning only when word_count < 1800.
    warnings = []
    hard_failures = []
    word_count = len(content.split())

    # ── Hard failures (post is discarded) ────────────────────────────────────

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

    # PATCH-3 FIX: removed the title-overlap hard failure that was here.
    # A quality intro naturally echoes the title's keywords — that is SEO
    # best practice, not a quality problem. We keep a WARNING at a much
    # higher threshold (0.95) so genuinely copy-pasted intros are still
    # flagged without discarding legitimate content.
    title_words = set(re.sub(r'[^\w\s]', '', title.lower()).split())
    title_words -= {'the', 'a', 'an'}
    if title_words and word_count < 1800:
        first_para_words = set(
            re.sub(r'[^\w\s]', '', content[:500].lower()).split()
        )
        title_overlap = len(title_words & first_para_words) / len(title_words)
        if title_overlap > 0.95:
            warnings.append(
                f"Opening section may be a near-verbatim restatement of the title "
                f"({title_overlap:.0%} title word overlap in first 500 chars). "
                "Consider a more specific, experience-driven opening paragraph."
            )

    # ── Warnings (logged; post still publishes) ───────────────────────────────

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

    if "```" not in content:
        warnings.append(
            "No code examples. Reduces substantive value for technical topics."
        )

    number_re = re.compile(
        r"\b(\d+%|\d+ms|\d+x\b|\$\d|\d+ req|\d+ min|p\d{2}|\d+,\d{3})"
    )
    if not number_re.search(content):
        warnings.append("No concrete numbers/metrics found.")

    version_re = re.compile(
        r'\b(Python|Node\.js|TypeScript|PostgreSQL|Redis|Django|FastAPI|React|'
        r'Next\.js|Docker|Kubernetes|Kafka|MySQL)\s+\d+[\.\d]*\b',
        re.IGNORECASE
    )
    if not version_re.search(content):
        warnings.append(
            "No version-pinned tool reference found (e.g. 'Python 3.13', 'Redis 7.2'). "
            "Version pins distinguish original from generic content."
        )

    if "frequently asked questions" not in content.lower() and "## faq" not in content.lower():
        warnings.append(
            "No FAQ section found. FAQ structured data improves AdSense eligibility signals."
        )

    if "|" not in content:
        warnings.append(
            "No markdown table found. A comparison table signals substantive content."
        )

    if "### About this article" not in content:
        warnings.append(
            "E-E-A-T author footer missing. Run inject_eeat_signals() before saving."
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
# PRE-FLIGHT SIMILARITY INDEX
# ─────────────────────────────────────────────────────────────────

_PREFLIGHT_CACHE_FILE = Path(".preflight_index.json")
_PREFLIGHT_CACHE_TTL_SECONDS = 3600
_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD = 0.55
_PREFLIGHT_MAX_RETRIES = 3


class PreFlightIndex:
    """
    Lightweight TF-IDF pre-flight similarity check.
    """

    def __init__(self, docs_dir: Path, cache_file: Path = _PREFLIGHT_CACHE_FILE):
        self.docs_dir = docs_dir
        self.cache_file = cache_file
        self._entries: List[Dict] = []
        self._vectorizer = None
        self._matrix = None
        self._loaded = False

    def load(self, force_rebuild: bool = False) -> None:
        if self._loaded and not force_rebuild:
            return
        try:
            if not force_rebuild and self._cache_is_fresh():
                self._load_from_cache()
            else:
                self._rebuild_from_docs()
                self._save_cache()
            self._fit_vectorizer()
            self._loaded = True
            print(
                f"  PreFlightIndex ready: {len(self._entries)} posts indexed.")
        except Exception as exc:
            print(f"  ⚠️  PreFlightIndex load failed (non-fatal): {exc}")
            self._entries = []
            self._loaded = True

    def is_duplicate(self, candidate: str) -> tuple:
        if not self._loaded:
            self.load()
        try:
            return self._cosine_check(candidate)
        except Exception as exc:
            print(
                f"  ⚠️  PreFlightIndex.is_duplicate error (non-fatal): {exc}")
            return False, "", 0.0

    def add_entry(self, slug: str, title: str, content: str) -> None:
        summary = self._make_summary(content)
        self._entries.append(
            {"slug": slug, "title": title, "summary": summary})
        try:
            self._fit_vectorizer()
            self._save_cache()
        except Exception as exc:
            print(f"  ⚠️  PreFlightIndex.add_entry failed (non-fatal): {exc}")

    def invalidate(self) -> None:
        self._loaded = False
        if self.cache_file.exists():
            self.cache_file.unlink(missing_ok=True)

    def _cache_is_fresh(self) -> bool:
        if not self.cache_file.exists():
            return False
        try:
            mtime = self.cache_file.stat().st_mtime
            age = datetime.now().timestamp() - mtime
            return age < _PREFLIGHT_CACHE_TTL_SECONDS
        except OSError:
            return False

    def _load_from_cache(self) -> None:
        with open(self.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._entries = data.get("entries", [])
        print(
            f"  PreFlightIndex: loaded {len(self._entries)} entries from cache.")

    def _rebuild_from_docs(self) -> None:
        self._entries = []
        if not self.docs_dir.exists():
            return
        for post_dir in self.docs_dir.iterdir():
            if not post_dir.is_dir() or post_dir.name == "static":
                continue
            post_json = post_dir / "post.json"
            if not post_json.exists():
                continue
            try:
                with open(post_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("title", "").strip()
                content = data.get("content", "")
                if title:
                    self._entries.append({
                        "slug": post_dir.name,
                        "title": title,
                        "summary": self._make_summary(content),
                    })
            except Exception:
                pass
        print(
            f"  PreFlightIndex: rebuilt {len(self._entries)} entries from docs/.")

    def _save_cache(self) -> None:
        # PATCH-4: atomic write — prevents cache corruption from concurrent runs
        import os as _os
        import tempfile as _tempfile

        payload = {
            "built_at": datetime.now().isoformat(),
            "entries": self._entries,
        }
        cache_dir = self.cache_file.parent
        try:
            fd, tmp_path = _tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            try:
                with _os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                # os.replace is atomic: readers see old file or new, never partial
                _os.replace(tmp_path, self.cache_file)
            except Exception:
                try:
                    _os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            print(
                f"  ⚠️  PreFlightIndex._save_cache atomic write failed: {exc}")

    def _make_summary(self, content: str, max_chars: int = 300) -> str:
        text = re.sub(r"```[\s\S]*?```", " ", content)
        text = re.sub(r"`[^`]+`", " ", text)
        text = re.sub(r"#{1,6}\s+", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"[*_]{1,3}", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    def _fit_vectorizer(self) -> None:
        if not self._entries:
            self._vectorizer = None
            self._matrix = None
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [
            f"{e['title']} {e['summary']}"
            for e in self._entries
        ]
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def _cosine_check(self, candidate: str) -> tuple:
        if self._vectorizer is None or self._matrix is None:
            return False, "", 0.0

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vec = self._vectorizer.transform([candidate])
        scores = cosine_similarity(vec, self._matrix).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_title = self._entries[best_idx]["title"]

        blocked = best_score >= _PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD
        return blocked, best_title, best_score


# ─────────────────────────────────────────────────────────────────
# BlogSystem
# ─────────────────────────────────────────────────────────────────


class BlogSystem:
    def __init__(self, config=None):
        if config is None:
            if os.path.exists("config.yaml"):
                with open("config.yaml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
        self.config = config
        self.output_dir = Path("./docs")
        self.output_dir.mkdir(exist_ok=True)

        self.og_dir = self.output_dir / "static" / "og"
        self.scripts_dir = Path("scripts")
        self.og_dir.mkdir(parents=True, exist_ok=True)

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

        self.preflight_index = PreFlightIndex(docs_dir=self.output_dir)

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

    def generate_og_images(self) -> bool:
        """
        Generate per-article Open Graph images (1200×630 PNG).

        Returns:
            bool: True if successful, False otherwise
        """

        if not PILLOW_AVAILABLE:
            print("⚠️  Pillow not installed. Skipping OG image generation.")
            return False

        script_path = self.scripts_dir / "generate_og_images.py"

        if not script_path.exists():
            print(f"⚠️  OG image script not found at {script_path}")
            return False

        base_url = self.config.get(
            "base_url", "https://kubaik.github.io").rstrip("/")

        print("\n" + "="*80)
        print("📸 Generating per-article OG images (1200×630 PNG)")
        print("="*80)

        try:
            cmd = [
                sys.executable,
                str(script_path),
                "--posts-dir", str(self.output_dir),
                "--output-dir", str(self.og_dir),
                "--base-url", base_url,
                "--patch-html",
            ]

            font_dir = Path("static/fonts")
            if font_dir.exists():
                cmd.extend(["--font-dir", str(font_dir)])

            print(f"\nRunning OG generation...\n")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                print(f"⚠️  OG image generation had issues:")
                if result.stderr:
                    print(result.stderr)
                return False

            og_files = list(self.og_dir.glob("*.png"))
            og_count = len(og_files)

            print(f"\n✅ OG image generation successful!")
            print(f"   Generated: {og_count} images")

            return True

        except subprocess.TimeoutExpired:
            print("❌ OG image generation timed out (>10 minutes)")
            return False
        except Exception as e:
            print(f"❌ OG image generation failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # STALE CONTENT REFRESH
    # ─────────────────────────────────────────────────────────────

    async def refresh_stale_posts(self, limit: int = 2) -> dict:
        from adsense_fixes.content_freshness import mark_stale_posts

        results = {
            "refreshed": [],
            "skipped": [],
            "errors": [],
        }

        stale_posts = mark_stale_posts(
            self.output_dir, days_threshold=STALE_THRESHOLD_DAYS
        )

        if not stale_posts:
            print("✅  No stale posts detected.")
            return results

        stale_posts.sort(
            key=lambda x: (
                0 if x['priority'] == 'high' else 1,
                -x['days_old'],
            )
        )

        print(
            f"\nFound {len(stale_posts)} stale post(s). "
            f"Refreshing top {limit}..."
        )

        for i, stale_post in enumerate(stale_posts[:limit]):
            slug = stale_post['slug']
            title = stale_post['title']
            days_old = stale_post['days_old']
            is_fast_decay = stale_post['fast_decay']

            print(
                f"\n[{i+1}/{min(limit, len(stale_posts))}] "
                f"Refreshing: {slug} "
                f"(title: {title}, {days_old} days old, "
                f"fast_decay={'yes' if is_fast_decay else 'no'})"
            )

            post_dir = self.output_dir / slug
            post_json = post_dir / "post.json"

            if not post_json.exists():
                msg = f"post.json not found for {slug}"
                print(f"  ⚠️  Skip: {msg}")
                results["skipped"].append(msg)
                continue

            try:
                with open(post_json, "r", encoding="utf-8") as f:
                    post_data = json.load(f)
            except Exception as e:
                msg = f"{slug}: failed to load post.json ({e})"
                print(f"  ❌  Error: {msg}")
                results["errors"].append(msg)
                continue

            original_title = post_data.get("title", "")
            original_content = post_data.get("content", "")
            seo_keywords = post_data.get("seo_keywords", [])

            if not original_content or len(original_content.split()) < 1000:
                msg = f"{slug}: content too short or empty"
                print(f"  ⚠️  Skip: {msg}")
                results["skipped"].append(msg)
                continue

            try:
                refreshed_content = await self._refresh_post_content(
                    original_title=original_title,
                    original_content=original_content,
                    seo_keywords=seo_keywords,
                    days_stale=days_old,
                    is_fast_decay=is_fast_decay,
                )
            except Exception as e:
                msg = f"{slug}: LLM refresh failed ({e})"
                print(f"  ❌  Error: {msg}")
                results["errors"].append(msg)
                continue

            refreshed_count = _count_words(refreshed_content)
            original_count = _count_words(original_content)

            if refreshed_count < MIN_ACCEPTABLE_WORDS:
                msg = (
                    f"{slug}: refreshed content too short "
                    f"({refreshed_count} < {MIN_ACCEPTABLE_WORDS} words)"
                )
                print(f"  ⚠️  Skip: {msg}")
                results["skipped"].append(msg)
                continue

            print(
                f"  ✓ LLM refresh complete: "
                f"{original_count} → {refreshed_count} words"
            )

            post_data["content"] = refreshed_content
            post_data["updated_at"] = datetime.now().isoformat()

            try:
                _inject_freshness_footer_inline(post_data)
                print(f"  ✓ Freshness footer updated")
            except Exception as e:
                print(
                    f"  ⚠️  Freshness footer update failed (non-fatal): {e}")

            try:
                with open(post_json, "w", encoding="utf-8") as f:
                    json.dump(post_data, f, indent=2, ensure_ascii=False)
                print(f"  ✓ post.json saved")

                with open(post_dir / "index.md", "w", encoding="utf-8") as f:
                    f.write(f"# {original_title}\n\n{refreshed_content}")
                print(f"  ✓ index.md saved")

                results["refreshed"].append(slug)
                print(f"  ✅  REFRESHED: {slug}")

            except Exception as e:
                msg = f"{slug}: failed to write files ({e})"
                print(f"  ❌  Error: {msg}")
                results["errors"].append(msg)

        return results

    async def _refresh_post_content(
        self,
        original_title: str,
        original_content: str,
        seo_keywords: list,
        days_stale: int,
        is_fast_decay: bool,
    ) -> str:
        # PATCH-5: removed redundant 400-word excerpt from the user message;
        # the system prompt already instructs the model to read full content carefully.
        keywords_str = ", ".join(seo_keywords[:8])

        decay_context = (
            "This is a FAST-DECAY technical topic (AI, LLM, cloud, Kubernetes, DevOps). "
            "Tool versions, API endpoints, and best practices may have shifted significantly."
        ) if is_fast_decay else (
            "This is a standard-decay topic. Core concepts are stable, but tool versions and "
            "examples should be modernized."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical content expert. Your job is to refresh an existing "
                    "blog post while preserving its SEO value and core message.\n\n"
                    "CONSTRAINTS (ABSOLUTE):\n"
                    "1. Keep the title EXACTLY as provided — do not change it\n"
                    "2. Keep the slug EXACTLY as provided — do not change it\n"
                    "3. Preserve all SEO keywords — they are embedded in the original content\n"
                    "4. Preserve the original article structure (headings, sections)\n"
                    "5. Preserve author voice and first-person anecdotes\n"
                    "6. Preserve all code examples, but update tool/library versions\n"
                    "7. Do not remove sections — only update facts, versions, and recommendations\n"
                    "8. Do not add new sections (no FAQ, no new deep-dives)\n"
                    "9. Preserve the E-E-A-T footer ('### About this article') — do NOT remove it\n"
                    "10. Current year is 2026 — update all year references and statistics\n\n"
                    "WHAT TO UPDATE:\n"
                    "- Tool versions: 'Python 3.9' → 'Python 3.13', 'Node 18' → 'Node 22 LTS'\n"
                    "- API endpoints: check if deprecated/changed\n"
                    "- Deprecation warnings: flag if library/tool mentioned is EOL\n"
                    "- Performance figures: note if 2024/2023 benchmarks now seem outdated\n"
                    "- Cost comparisons: update SaaS pricing if known to have changed\n"
                    "- Best practices: modernize patterns (e.g., callbacks → async/await)\n"
                    "- Security guidance: update if new vulnerabilities/mitigations exist\n"
                    "- Framework features: if the library added major features, mention them\n"
                    "- Alternative tools: note if landscape changed (new competitors, acquisitions)\n\n"
                    "WHAT NOT TO CHANGE:\n"
                    "- The core point/angle of the article\n"
                    "- The title (word-for-word)\n"
                    "- The slug\n"
                    "- The author persona or voice\n"
                    "- The section headings (only update content within sections)\n"
                    "- Any copyright/attribution statements\n"
                    "- The 'About this article' footer\n\n"
                    "PROCESS:\n"
                    "1. Read the original content carefully\n"
                    "2. Identify outdated version numbers, API endpoints, tool recommendations\n"
                    "3. Update in-place: replace old references with 2026 equivalents\n"
                    "4. If unsure about a fact, add: 'As of 2026, [claim]. (Verify against latest.)'\n"
                    "5. Keep all prose, examples, and structure identical except factual updates\n"
                    "6. Return the COMPLETE refreshed article (all sections, all content)\n"
                    "7. Preserve all markdown formatting, code blocks, tables, and emphasis\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n"
                    f"Post age: {days_stale} days old\n"
                    f"Decay type: {decay_context}\n"
                    f"Title (DO NOT CHANGE): {original_title}\n"
                    f"SEO Keywords (preserve in content): {keywords_str}\n\n"
                    f"FULL ORIGINAL CONTENT:\n"
                    f"{original_content}\n\n"
                    f"TASK:\n"
                    f"Refresh this post by updating:\n"
                    f"1. Tool/library/framework versions to their 2026 equivalents\n"
                    f"2. Deprecated APIs or endpoints\n"
                    f"3. Security/best-practice guidance (if applicable)\n"
                    f"4. Cost figures or pricing comparisons\n"
                    f"5. Benchmark numbers (latency, throughput, etc.)\n"
                    f"6. Outdated statistics or market data\n\n"
                    f"Do NOT:\n"
                    f"- Change the title\n"
                    f"- Remove sections\n"
                    f"- Add new sections or FAQ\n"
                    f"- Alter the voice or author anecdotes\n"
                    f"- Remove or modify the 'About this article' footer\n"
                    f"- Change section headings\n"
                    f"- Change code examples unless the syntax is deprecated\n\n"
                    f"Return the COMPLETE refreshed article (every section, every paragraph)."
                ),
            },
        ]

        return await self._call_api_with_fallback(messages, max_tokens=6500)

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
            "X-Title": self.config.get("site_name", "Kubai Kevin"),
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

    async def generate_blog_post(self, topic: str, keywords: List[str] = None) -> "BlogPost":
        if not self.api_key:
            print("No API keys configured. Using local template content.")
            return self._generate_fallback_post(topic)

        self.preflight_index.load()

        current_topic = topic
        current_keywords = keywords
        preflight_attempts = 0

        SEP = "─" * 60
        print(f"\n{SEP}")
        print(f"PRE-FLIGHT CHECK for topic: '{current_topic}'")

        while preflight_attempts < _PREFLIGHT_MAX_RETRIES:
            blocked, match_title, pf_score = self.preflight_index.is_duplicate(
                current_topic
            )
            if not blocked:
                print(
                    f"  ✅ Pre-flight OK (best match score {pf_score:.2f} < "
                    f"{_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD}). Proceeding."
                )
                break

            preflight_attempts += 1
            print(
                f"  ⚠️  Pre-flight BLOCKED (attempt {preflight_attempts}/{_PREFLIGHT_MAX_RETRIES}):\n"
                f"     Score {pf_score:.2f} ≥ {_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD} "
                f"vs '{match_title}'\n"
                f"     Asking LLM for a distinct topic…"
            )

            if preflight_attempts >= _PREFLIGHT_MAX_RETRIES:
                print(
                    f"  🛑 Pre-flight max retries ({_PREFLIGHT_MAX_RETRIES}) reached. "
                    "Proceeding with last candidate — post-generation duplicate "
                    "checks will still apply."
                )
                break

            try:
                current_topic = await self._ask_llm_for_distinct_topic(
                    blocked_topic=current_topic,
                    similar_title=match_title,
                    similarity_score=pf_score,
                )
                current_keywords = None
                print(f"  LLM suggested: '{current_topic}'")
            except Exception as exc:
                print(
                    f"  LLM topic suggestion failed ({exc}). Continuing with current topic.")
                break

        print(SEP)

        attempted_topics: List[str] = []

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
                post.prewritten_tweet = "\n\n".join(parts)

                print(
                    f"Bundle tweet assembled: {len(post.prewritten_tweet)} raw chars "
                    f"(effective ~{len(bundle_tweet) + fixed_cost + tags_cost + bait_cost} after t.co)\n"
                    f"  Body    : {len(bundle_tweet)} chars (budget {body_budget})\n"
                    f"  Hashtags: {hashtag_str or '(none)'}\n"
                    f"  Bait    : {reply_bait if reply_bait else '(dropped)'}"
                )
            else:
                post.prewritten_tweet = ""
                print("Note: no tweet_text in bundle — template fallback will be used.")

            return post

        raise InsufficientContentError(
            f"Exhausted {MAX_GENERATION_ATTEMPTS} generation attempts without "
            f"producing adequate content. No post has been saved."
        )

    # ─────────────────────────────────────────────────────────────
    # LLM-BASED DISTINCT TOPIC SUGGESTER
    # ─────────────────────────────────────────────────────────────

    async def _ask_llm_for_distinct_topic(
        self,
        blocked_topic: str,
        similar_title: str,
        similarity_score: float,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical blog editor. Your only job right now is to "
                    "propose a single, distinct blog topic. Respond with ONLY the topic "
                    "— no quotes, no explanation, no JSON, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The topic '{blocked_topic}' is too similar to an existing post "
                    f"titled '{similar_title}' (similarity score: {similarity_score:.0%}).\n\n"
                    "Propose ONE new blog topic that:\n"
                    "- Covers a meaningfully different angle, sub-topic, or technology\n"
                    "- Is within the same broad domain (software engineering / developer tools / "
                    "AI / backend / career)\n"
                    "- Is specific enough to generate 2000+ words of original content\n"
                    "- Does NOT repeat the existing post's core subject\n\n"
                    "Respond with ONLY the topic text."
                ),
            },
        ]
        raw = await self._call_api_with_fallback(messages, max_tokens=80)
        new_topic = raw.strip().strip('"').strip("'").strip()
        return new_topic if new_topic else blocked_topic

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
        reading_time = max(1, round(word_count / 200))

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

        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        post_data = post.to_dict()
        post_data['word_count'] = word_count
        post_data['reading_time_minutes'] = reading_time
        post_data['has_code'] = '```' in post.content
        post_data['has_table'] = '|' in post.content

        if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags:
            post_data['twitter_hashtags'] = post.twitter_hashtags

        if hasattr(post, 'prewritten_tweet') and post.prewritten_tweet:
            post_data['prewritten_tweet'] = post.prewritten_tweet

        with open(post_dir / "post.json", "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=2, ensure_ascii=False)

        with open(post_dir / "index.md", "w", encoding="utf-8") as f:
            f.write(f"# {post.title}\n\n{post.content}")

        try:
            self.preflight_index.add_entry(
                slug=post.slug,
                title=post.title,
                content=post.content,
            )
        except Exception as exc:
            print(
                f"  ⚠️  PreFlightIndex post-save update failed (non-fatal): {exc}")

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
# Freshness footer helper (works on dict, used by refresh-stale)
# ─────────────────────────────────────────────────────────────────

def _inject_freshness_footer_inline(post_data: dict) -> None:
    if not post_data.get('content', ''):
        return

    today_str = datetime.now().strftime('%B %d, %Y')
    reviewed_pattern = r'(\*\*Last reviewed:\*\*\s*)([^\n]+)'

    post_data['content'] = re.sub(
        reviewed_pattern,
        lambda m: f"{m.group(1)}{today_str}",
        post_data['content'],
    )


# ─────────────────────────────────────────────────────────────────
# TOPIC PICKER
# ─────────────────────────────────────────────────────────────────

def pick_next_topic(
    config_path: str = "config.yaml",
    history_file: str = ".used_topics.json",
    preflight_index: "PreFlightIndex | None" = None,
) -> str:
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
            print(
                f"Skipped {len(skipped)} topic(s) already covered (Jaccard):")
            for topic, match, score in skipped:
                print(f"  '{topic}' ≈ '{match}' ({score:.0%})")

        if safe_available:
            available = safe_available
        else:
            print("All available topics covered (Jaccard). Resetting.")
            available = topics
            used = []

    if available:
        if preflight_index is None:
            preflight_index = PreFlightIndex(docs_dir=docs_dir)
        preflight_index.load()

        pf_safe, pf_skipped = [], []
        for candidate in available:
            blocked, match_title, pf_score = preflight_index.is_duplicate(
                candidate)
            if blocked:
                pf_skipped.append((candidate, match_title, pf_score))
            else:
                pf_safe.append(candidate)

        if pf_skipped:
            print(
                f"Skipped {len(pf_skipped)} topic(s) already covered (TF-IDF pre-flight):")
            for t, m, s in pf_skipped:
                print(f"  '{t}' ≈ '{m}' ({s:.0%})")

        if pf_safe:
            available = pf_safe
        else:
            print(
                "All remaining topics are TF-IDF near-duplicates. "
                "Falling back to Jaccard-safe list."
            )

    topic = random.choice(available)
    used.append(topic)

    with open(history_file, "w") as f:
        json.dump(used, f, indent=2)

    print(f"Selected topic: {topic}")
    return topic


# ─────────────────────────────────────────────────────────────────
# CONFIG INITIALISER
# ─────────────────────────────────────────────────────────────────

def create_sample_config(config_path: str = "config.yaml"):
    """
    Safe idempotent init — never overwrites keys the user has already set.

    Behaviour:
    - First run (no config.yaml): writes the full default config.
    - Subsequent runs: reads the existing file, adds any MISSING keys,
      appends any NEW topics not already in the list, then writes back.
      Every key the user has already customised is left untouched.
    """
    CONFIG_FILE = config_path

    # ── Load existing config (empty dict on first run) ──────────────────────
    existing: dict = {}
    is_new_file = not os.path.exists(CONFIG_FILE)
    if not is_new_file:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
                existing = yaml.safe_load(_f) or {}
            print(
                f"  Found existing {CONFIG_FILE} — merging new defaults only.")
        except Exception as _e:
            print(f"  Warning: could not read {CONFIG_FILE} ({_e}). "
                  "Treating as new file.")
            existing = {}
            is_new_file = True

    # ── Scalar defaults (only written when key is absent) ───────────────────
    # Keys that contain real credentials are intentionally left as empty
    # strings so a first-run user sees clearly what they need to fill in,
    # while a returning user never has their live values clobbered.
    SCALAR_DEFAULTS = {
        "site_name":               "Kubai Kevin",
        "site_description": (
            "Practical backend engineering, AI tooling, and developer career "
            "advice by Kubai Kevin — 10+ years building production systems."
        ),
        "base_url":                "https://kubaik.github.io",
        "base_path":               "",
        "amazon_affiliate_tag":    "aiblogcontent-20",
        "google_analytics_id":     "",
        "google_adsense_id":       "",
        # Must be the HTML-meta verification token from Search Console
        # (Settings → Ownership verification → HTML tag), NOT an API key.
        "google_search_console_key": "",
        "hook_style":              "auto",
    }

    changed_keys: list[str] = []
    for key, default in SCALAR_DEFAULTS.items():
        if key not in existing:
            existing[key] = default
            changed_keys.append(key)

    # ── google_search_console_key: silently clear accidental API keys ────────
    gsc = existing.get("google_search_console_key", "")
    if isinstance(gsc, str) and gsc.startswith("AIza"):
        print(f"  ⚠️  google_search_console_key looks like a Google API key "
              f"(starts with 'AIza'). Clearing it — paste the HTML-meta "
              f"verification token from Search Console instead.")
        existing["google_search_console_key"] = ""
        changed_keys.append("google_search_console_key (cleared bad value)")

    # ── social_accounts: merge sub-keys, never overwrite existing values ─────
    social_defaults = {
        "twitter":  "https://twitter.com/KubaiKevin",
        "linkedin": "your-linkedin-page",
        "facebook": "your-facebook-page",
    }
    if "social_accounts" not in existing:
        existing["social_accounts"] = social_defaults
        changed_keys.append("social_accounts")
    else:
        for k, v in social_defaults.items():
            if k not in existing["social_accounts"]:
                existing["social_accounts"][k] = v
                changed_keys.append(f"social_accounts.{k}")

    # ── adsense_slots: add placeholder block when absent ─────────────────────
    if "adsense_slots" not in existing:
        existing["adsense_slots"] = {
            # Uncomment and paste real slot IDs once AdSense approves the site.
            # "header": "",
            # "inline": "",
            # "middle": "",
            # "footer": "",
        }
        changed_keys.append("adsense_slots")

    config = existing  # alias for clarity below
    # ── content_topics: append only topics not already present ───────────────
    NEW_TOPICS = [
        # ── AI & LLM Engineering ──────────────────────────────────────────────────
        "Why Claude 4 and GPT-5 changed how I structure prompts for production systems",
        "MCP servers in 2026: the protocol that quietly became infrastructure",
        "Agentic coding in 2026: what Claude Code actually gets right after a year of daily use",
        "How I built a self-healing deployment pipeline using AI agents",
        "RAG is not enough: why hybrid search + reranking is the 2026 baseline",
        "Fine-tuning small models vs prompting large ones: the 2026 cost-accuracy tradeoff",
        "AI code review pipelines: how teams are replacing PR checklists with agents",
        "The context window arms race: what 1M token models actually change in practice",
        "Why most AI agents in production still need a human in the loop",
        "How I use AI to maintain legacy codebases nobody wants to touch",
        "Tool-calling patterns that work in production vs the ones that break under load",
        "Structured outputs from LLMs: why JSON mode is not enough for real workflows",
        "Evaluating LLM output quality at scale: the metrics that actually matter",
        "How AI observability is different from traditional application monitoring",
        "Building personal AI assistants for developer workflows without vendor lock-in",
        "The hidden costs of running AI agents in production: tokens, retries, and timeouts",
        "How prompt injection attacks work and why your AI product is probably vulnerable",
        "AI-assisted incident response: what works and what makes oncall worse",
        "Cursor vs Windsurf vs Claude Code in 2026: a real daily-use comparison",
        "Why vibe coding works for MVPs and fails for anything you need to maintain",
        "How AI pair programming changes code ownership and review responsibility",
        "Model routing in 2026: how to pick the right LLM for each task automatically",
        "Local LLMs in 2026: the models that finally run usably on a MacBook Pro",
        "AI memory systems: giving your agent context that persists across sessions",
        "How multi-agent orchestration frameworks compare in 2026: LangGraph vs CrewAI vs custom",

        # ── Backend & Databases ───────────────────────────────────────────────────
        "Postgres in 2026: the features that replaced three separate tools in our stack",
        "How pgvector changed our architecture: replacing Pinecone with what we already had",
        "Serverless databases in 2026: Neon vs PlanetScale vs Turso after real production use",
        "Why async Python still surprises people and how to stop getting burned by it",
        "Edge-native backends in 2026: what changes when your API runs in 50 locations",
        "gRPC vs REST vs tRPC in 2026: which one I reach for and why",
        "How I migrated a monolith to services in 2026 without a big-bang rewrite",
        "Database branching workflows: how Neon and PlanetScale changed how I develop locally",
        "The ORM debate in 2026: Prisma vs Drizzle vs SQLAlchemy for new projects",
        "Durable execution in 2026: why Temporal and Inngest are replacing custom job queues",
        "API design in 2026: what GraphQL, REST, and tRPC each still do better than the others",
        "How HTMX changed my stack and what I gave up to get there",
        "Python 3.13 in production: the free-threaded GIL change and what it means for web apps",
        "Building webhook systems in 2026: delivery guarantees, retries, and idempotency",
        "How to design multi-region backends without triple the infrastructure cost",
        "The background job queue landscape in 2026: what replaced Celery in my stack",
        "OpenTelemetry in 2026: the standard that finally stuck and how to actually use it",
        "How I handle database schema migrations in a zero-downtime deployment pipeline",
        "Soft deletes at scale: why the simple pattern breaks and what to do instead",
        "Building audit logs that satisfy compliance requirements without killing performance",

        # ── Infrastructure & Platform Engineering ─────────────────────────────────
        "Platform engineering in 2026: what internal developer platforms look like at different company sizes",
        "How AI is changing infrastructure: automated capacity planning, anomaly detection, and runbooks",
        "FinOps in 2026: the AWS cost levers that actually move the needle for mid-size teams",
        "Kubernetes in 2026: what teams are offloading to managed platforms and what they keep",
        "How I replaced a $3k/month Kubernetes cluster with a $200/month alternative",
        "GitOps in 2026: what ArgoCD and Flux look like in teams that have been using them for two years",
        "Pulumi vs Terraform vs OpenTofu in 2026: the infrastructure-as-code landscape after the HashiCorp fork",
        "How container builds changed in 2026: BuildKit, depot.dev, and why CI pipelines got faster",
        "Observability in 2026: why the three pillars model is being replaced and what comes next",
        "How I built a production deployment pipeline for a solo project that costs under $20/month",
        "On-call in 2026: how AI alert triage is changing what wakes engineers up at night",
        "Graviton4 and Ampere in 2026: the ARM migration decision for production workloads",
        "How service mesh adoption actually went: what Istio and Cilium look like in real production",
        "Cloud egress costs in 2026: the architecture decisions that save thousands per month",
        "How I use Cloudflare Workers to handle 80% of traffic before it hits my origin server",

        # ── Security ──────────────────────────────────────────────────────────────
        "Supply chain security in 2026: how SLSA and Sigstore became baseline requirements",
        "How AI is being used to find vulnerabilities and what that means for defenders",
        "Prompt injection in production AI systems: the attack surface most teams are ignoring",
        "Zero-trust in practice for small teams: what you can implement without an enterprise budget",
        "How passkeys changed authentication and what that means for the apps you're building",
        "Secrets management in 2026: the approaches that replaced .env files in serious projects",
        "How to implement least-privilege IAM in AWS without spending a week on policies",
        "The OWASP Top 10 for LLM applications: the vulnerabilities specific to AI-powered apps",
        "How post-quantum cryptography affects the TLS stack you're running today",
        "SBOM requirements in 2026: what software bill of materials means for your release process",
        "How I run automated security scanning in CI without drowning in false positives",
        "API security in 2026: the attack patterns that are becoming more common and how to block them",

        # ── Frontend & Full-Stack ─────────────────────────────────────────────────
        "React 19 in production: what the compiler actually changed about how I write components",
        "Next.js 15 vs Remix vs SvelteKit in 2026: a framework comparison after shipping real projects",
        "How AI UI generation tools changed my frontend workflow (and what they still cannot do)",
        "Web performance in 2026: the Core Web Vitals update and what changed about INP",
        "How I use server components to cut client bundle size without rewriting everything",
        "CSS in 2026: container queries, @layer, and the features that finally retired my utility framework",
        "Building accessible AI chat interfaces: the WCAG requirements most teams miss",
        "How Signals changed state management and whether it matters outside of frameworks",
        "The islands architecture in 2026: when Astro and partial hydration are the right choice",
        "TypeScript 5.x features that changed how I model domain logic",
        "How I test frontend code in 2026: Playwright, Vitest, and why I dropped Cypress",
        "Progressive Web Apps in 2026: what the browser APIs finally made possible",

        # ── Career & Remote Work ──────────────────────────────────────────────────
        "Tech salaries in 2026: what the market correction settled at and where it's growing",
        "How AI changed what hiring managers are looking for in engineering interviews",
        "The skills that protect your salary when AI automates junior developer tasks",
        "Remote work in 2026: which companies pulled back and which doubled down",
        "How to negotiate compensation in 2026 when AI can do parts of your job description",
        "Staff engineer vs engineering manager in 2026: how the roles evolved with AI tooling",
        "How to build a portfolio that gets you hired in a market where everyone has AI-assisted projects",
        "The engineering interview has changed: what 2026 technical screens actually look like",
        "How developers in Nairobi and Lagos are landing $5k/month remote roles in 2026",
        "From bootcamp to $120k: the realistic 2026 timeline and what actually fills the gap",
        "Freelance developer rates in 2026: what the AI productivity tools did to pricing",
        "How to stay technically relevant when the tools change faster than you can learn them",
        "Developer burnout in the AI era: why output pressure increased even as tools improved",
        "Building a second income stream as a developer in 2026 without building a SaaS",
        "How to pass AI-era technical interviews: the new formats replacing LeetCode",
        "Andela, Toptal, Arc, and Contra in 2026: which platforms still work for African developers",

        # ── African Tech & Emerging Markets ──────────────────────────────────────
        "M-Pesa Daraja 2.0 in 2026: what changed and how to migrate your integration",
        "How to build payment systems that work across Kenya, Nigeria, and Ghana without three separate integrations",
        "Building for 4G-as-baseline users in 2026: what changed when Starlink reached East Africa",
        "AI tools built for African developers in 2026: what actually exists now",
        "How African fintech regulations changed API design requirements in 2026",
        "The infrastructure stack for a Nairobi-based SaaS in 2026: a real cost breakdown",
        "How to build offline-first apps for markets where connectivity is still unreliable",
        "USSD is not dead: why feature phone interfaces still matter for African fintech in 2026",
        "How East African developers are using AI tools to compete with teams in higher-cost markets",
        "Building for mobile money reconciliation: the edge cases that Paystack and Flutterwave don't document",

        # ── System Design & Architecture ─────────────────────────────────────────
        "System design in 2026: how AI assistants changed the interview and the real job",
        "How to design AI-native applications: architecture patterns that didn't exist three years ago",
        "Event-driven architecture in 2026: what Kafka, Redpanda, and NATS each do best",
        "How I design for LLM latency: the patterns that keep AI features feeling fast",
        "Designing multi-tenant SaaS in 2026: row-level security vs schema-per-tenant vs database-per-tenant",
        "The data pipeline stack in 2026: what replaced Airflow for teams that couldn't afford the overhead",
        "How to design a search system in 2026: semantic, keyword, and hybrid approaches compared",
        "Building for eventual consistency: the real-world patterns behind systems that stay up",
        "How feature flag systems evolved into full experimentation platforms",
        "Designing notification systems that work across push, email, SMS, and WhatsApp",

        # ── Indie Hacking & SaaS ──────────────────────────────────────────────────
        "How AI changed the economics of building a solo SaaS in 2026",
        "Micro-SaaS in 2026: the niches that are working and the ones that got commoditised by AI",
        "How I built and launched a SaaS in 6 weeks using AI tools for 80% of the code",
        "Pricing a developer tool in 2026: what the market teaches you fast",
        "How I got to $5k MRR without a marketing team or a growth budget",
        "AI wrapper businesses in 2026: why most failed and the ones that survived",
        "How to pick a SaaS niche in 2026 when AI is disrupting entire software categories",
        "Stripe vs Lemon Squeezy vs Paddle in 2026: which one I use and why",
        "Building developer tools as a business: what the AI era changed about the sales cycle",
        "How to build in public in 2026 without the performance burning you out",

        # ── Software Craft ────────────────────────────────────────────────────────
        "Code review in the AI era: what the process looks like when half the code is generated",
        "How to maintain a codebase where 40% of the code was written by an AI",
        "Technical debt in 2026: the new category created by AI-generated code",
        "How I write tests for AI-assisted code without testing the AI",
        "Documentation in 2026: how AI changed what humans still need to write",
        "The engineering principles I updated after AI tools changed my daily workflow",
        "How to do a postmortem on an AI agent failure: it's different from a regular incident",
        "Pair programming with AI: how it changed collaboration on my team",
        "How to onboard a developer to a partially AI-generated codebase",
        "Feature flags in 2026: how they became the backbone of AI rollout strategies",
    ]

    existing_topics: list = config.get("content_topics") or []
    existing_topic_set = set(existing_topics)
    appended_topics = [t for t in NEW_TOPICS if t not in existing_topic_set]
    config["content_topics"] = existing_topics + appended_topics
    if appended_topics:
        changed_keys.append(
            f"content_topics (+{len(appended_topics)} new, "
            f"{len(existing_topics)} preserved)"
        )

    # ── Write back ───────────────────────────────────────────────────────────
    with open(CONFIG_FILE, "w", encoding="utf-8") as _f:
        yaml.dump(config, _f, default_flow_style=False,
                  indent=2, allow_unicode=True)

    if is_new_file:
        print(f"Created {CONFIG_FILE} with default configuration.")
    elif changed_keys:
        print(f"Updated {CONFIG_FILE} — added missing keys:")
        for k in changed_keys:
            print(f"  + {k}")
    else:
        print(f"{CONFIG_FILE} is already up to date — nothing changed.")

    print(
        "\nRequired GitHub secrets: GROQ_API_KEY, OPENROUTER_API_KEY, "
        "CEREBRAS_API_KEY, MISTRAL_API_KEY, NVIDIA_API_KEY, GEMINI_API_KEY, "
        "BLOGGITHUB_TOKEN, CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID"
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
            cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
            create_sample_config(config_path=cfg_path)
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
                topic = pick_next_topic(
                    preflight_index=blog_system.preflight_index
                )
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

            try:
                guard = SimilarityGuard(docs_dir=blog_system.output_dir)
                sim_result = guard.check(blog_post)
                if sim_result.is_blocked:
                    print(f"\n🛑 SIMILARITY BLOCK: {sim_result.reason}\n")
                    sys.exit(1)
                for warning in sim_result.warnings:
                    print(f"  ⚠️  Similarity: {warning}")
            except Exception as sim_err:
                print(f"  ⚠️  SimilarityGuard failed (non-fatal): {sim_err}")

            quality_warnings, hard_failures = _validate_content_quality(
                blog_post.content, blog_post.title
            )

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
            inject_freshness_footer(blog_post)

            try:
                injected_imgs = inject_alt_text(blog_post)
                if injected_imgs:
                    print(f"  🖼  {injected_imgs} image alt text(s) injected.")
            except Exception as e:
                print(f"  ⚠️  Alt text injection failed (non-fatal): {e}")

            try:
                posts_index = build_posts_index(blog_system.output_dir)
                base_path = config.get("base_path", "")
                inject_internal_links(
                    blog_post, posts_index, base_path=base_path)
            except Exception as e:
                print(f"  ⚠️  Internal link injection failed (non-fatal): {e}")

            try:
                removed_links = validate_post_links(
                    blog_post, blog_system.output_dir)
                if removed_links:
                    print(f"  🔗 Link validator removed {len(removed_links)} unresolvable link(s): "
                          f"{', '.join(removed_links)}")
            except Exception as e:
                print(f"  ⚠️  Link validator failed (non-fatal): {e}")

            try:
                canon_issues = validate_canonical(
                    blog_post, config.get('base_url', ''))
                for issue in canon_issues:
                    print(f"  ⚠️  Canonical: {issue}")
            except Exception as e:
                print(f"  ⚠️  Canonical validation failed (non-fatal): {e}")

            blog_system.save_post(blog_post)

            try:
                generate_og_card(
                    blog_post,
                    output_dir=blog_system.output_dir,
                    site_name=config.get('site_name', 'Kubai Kevin'),
                )
            except Exception as e:
                print(f"  ⚠️  OG card generation failed (non-fatal): {e}")

            try:
                guard.update_index(blog_post)
            except Exception:
                pass

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

            blog_system = BlogSystem(config)
            success = blog_system.generate_og_images()

            if not success:
                print("⚠️  WARNING: OG image generation had issues (non-fatal)")
                # Continue anyway - OG generation is optional but recommended

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

            success = blog_system.generate_og_images()

            if not success:
                print("⚠️  WARNING: OG image generation had issues (non-fatal)")
                # Continue anyway - OG generation is optional but recommended

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

        elif mode == "refresh-stale":
            limit = 2
            args = sys.argv[2:]
            for i, arg in enumerate(args):
                if arg == "--limit" and i + 1 < len(args):
                    try:
                        limit = int(args[i + 1])
                    except (ValueError, IndexError):
                        limit = 2

            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            blog_system = BlogSystem(config)
            refresh_results = asyncio.run(
                blog_system.refresh_stale_posts(limit=limit)
            )

            print("\n" + "=" * 70)
            print("REFRESH RESULTS")
            print("=" * 70)
            print(f"Refreshed : {len(refresh_results['refreshed'])} posts")
            if refresh_results['refreshed']:
                for slug in refresh_results['refreshed']:
                    print(f"  ✓ {slug}")

            if refresh_results['skipped']:
                print(f"\nSkipped   : {len(refresh_results['skipped'])} posts")
                for reason in refresh_results['skipped']:
                    print(f"  - {reason}")

            if refresh_results['errors']:
                print(f"\nErrors    : {len(refresh_results['errors'])} posts")
                for error in refresh_results['errors']:
                    print(f"  ✗ {error}")

            print("=" * 70 + "\n")

            if refresh_results['refreshed']:
                print(f"has_refreshed=true")
                print(
                    f"refreshed_list={','.join(refresh_results['refreshed'])}")
            else:
                print(f"has_refreshed=false")

            if refresh_results['refreshed']:
                StaticSiteGenerator(blog_system).generate_site()
                print("Site rebuilt after stale-post refresh.")

        elif mode == "audit-links":
            from adsense_fixes.link_validator import audit_all_internal_links
            report = audit_all_internal_links(Path('./docs'))
            print(report)

        elif mode == "audit-slugs":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            report = audit_duplicate_slugs(Path('./docs'))
            print(report)

        elif mode == "audit-freshness":
            from adsense_fixes.content_freshness import stale_report
            print(stale_report(Path('./docs')))
            print()
            print(get_publishing_schedule_status(Path('./docs')))

        elif mode == "velocity":
            vc = VelocityController()
            subcmd = sys.argv[2] if len(sys.argv) > 2 else "status"
            if subcmd == "status":
                print(
                    f"Today: {vc.today_count()}/{vc.effective_limit()} posts published")
            elif subcmd == "reset":
                Path(".publish_velocity.json").unlink(missing_ok=True)
                print("Velocity counter reset.")
            else:
                print("Usage: python blog_system.py velocity [status|reset]")

        elif mode == "preflight-rebuild":
            docs_dir = Path("./docs")
            idx = PreFlightIndex(docs_dir=docs_dir)
            idx.load(force_rebuild=True)
            print(
                f"Pre-flight index rebuilt: {len(idx._entries)} posts indexed.")
            print(f"Cache written to: {idx.cache_file}")

        elif mode == "preflight-check":
            if len(sys.argv) < 3:
                print("Usage: python blog_system.py preflight-check <topic>")
                sys.exit(1)
            candidate = " ".join(sys.argv[2:])
            docs_dir = Path("./docs")
            idx = PreFlightIndex(docs_dir=docs_dir)
            idx.load()
            blocked, match_title, score = idx.is_duplicate(candidate)
            status = "BLOCKED" if blocked else "OK"
            print(f"Topic   : {candidate}")
            print(f"Status  : {status}")
            print(
                f"Score   : {score:.2f} (threshold {_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD})")
            if match_title:
                print(f"Nearest : {match_title}")

        else:
            print(
                "Usage: python blog_system.py [init|auto|build|cleanup|audit|purge|"
                "debug|social|test-twitter|dedup|fix-descriptions|refresh-stale|"
                "audit-links|audit-slugs|audit-freshness|velocity|"
                "preflight-rebuild|preflight-check]"
            )

    else:
        print("AI Blog System — Usage: python blog_system.py [command]")
        print("Commands: init | auto | build | cleanup | audit | purge | debug | social | "
              "test-twitter | dedup | fix-descriptions | refresh-stale | audit-links | "
              "audit-slugs | audit-freshness | velocity | preflight-rebuild | preflight-check")
