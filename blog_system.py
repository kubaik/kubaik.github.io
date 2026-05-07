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
MIN_WORD_COUNT = 2000
MIN_WORD_PURGE = 1500


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

    outcome = _extract_numbers(text)
    keyword = _extract_topic_phrase(title, max_words=4)

    benefit = ""
    benefit_patterns = [
        r'so (?:you|your team) can ([^.!?]{10,60})',
        r'without ([^.!?]{10,50})',
        r'that (?:lets?|allows?) you ([^.!?]{10,50})',
        r'to (?:help you\s+)?([a-z][^.!?]{10,50})',
    ]
    for pat in benefit_patterns:
        m = re.search(pat, text[:600], re.IGNORECASE)
        if m:
            benefit = m.group(1).strip().rstrip('.,;:')
            break

    if outcome and keyword and benefit:
        candidate = f"{outcome} with {keyword} — {benefit}."
        if len(candidate) <= max_len:
            return candidate
        candidate = f"{outcome} with {keyword}."
        if len(candidate) <= max_len:
            return candidate

    if outcome and keyword:
        candidate = f"{keyword}: {outcome}. Practical guide with real examples."
        if len(candidate) <= max_len:
            return candidate

    if keyword and benefit:
        candidate = f"{keyword} — {benefit}. Includes code, benchmarks, and common mistakes."
        if len(candidate) <= max_len:
            return candidate

    sentences = re.split(r'(?<=[.!?])\s+', text)
    excerpt = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 40:
            excerpt = sentence
            break

    if not excerpt:
        excerpt = text[:max_len]

    if len(excerpt) > max_len:
        excerpt = excerpt[:max_len].rsplit(" ", 1)[0].rstrip(".,;:")
        excerpt += "…"

    return excerpt


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
# Personal intro injection
# ─────────────────────────────────────────────────────────────────

_PERSONAL_INTROS = [
    (
        "This took me about three days to figure out properly. Most of the answers "
        "I found online were either outdated or skipped the parts that actually matter in production. "
        "Here's what I learned."
    ),
    (
        "A colleague asked me about this last week and I realised I couldn't explain it cleanly. "
        "Writing this post forced me to think it through properly — which is usually how it goes."
    ),
    (
        "I've seen this done wrong in more codebases than I can count, including my own early work. "
        "This is the post I wish I'd had when I started."
    ),
    (
        "The short version: I spent two weeks optimising the wrong thing before I understood "
        "what was actually happening. The longer version is below."
    ),
    (
        "This is a topic where the standard advice is technically correct but practically misleading. "
        "Here's the fuller picture, based on what I've seen work at scale."
    ),
    (
        "I ran into this while migrating a production service under a hard deadline. "
        "The official docs covered the happy path well. This post covers everything else."
    ),
    (
        "I've answered versions of this question in Slack, code reviews, and one-on-ones "
        "more times than I can count. Writing it down properly felt overdue."
    ),
    (
        "The thing that frustrated me most when learning this was that every tutorial "
        "assumed a clean slate. Real systems never are. Here's how it actually goes."
    ),
]


def inject_personal_intro(post, topic: str) -> None:
    idx = int(hashlib.md5(topic.encode()).hexdigest(),
              16) % len(_PERSONAL_INTROS)
    intro = _PERSONAL_INTROS[idx]
    if intro[:30] not in post.content:
        post.content = f"{intro}\n\n{post.content}"


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

        try:
            print(f"Generating content for: {topic}")
            existing_titles = _load_existing_titles(self.output_dir)

            bundle = await self._generate_full_bundle(topic, keywords, existing_titles)

            title = bundle["title"].strip().strip('"')
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

            if not keywords:
                keywords = seo_keywords

            word_count = _count_words(content)
            print(f"Generated content: {word_count} words")

            if word_count < MIN_WORD_COUNT:
                print(f"Content short ({word_count} words). Expanding once...")
                content = await self._expand_content(content, title, topic)
                word_count = _count_words(content)
                print(f"After expansion: {word_count} words")

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
                seo_keywords, topic=topic, title=title, max_hashtags=5)
            print(f"Hashtags selected: {', '.join(hashtags)}")

            post.tags = list(set(post.tags + hashtags))[:15]
            post.seo_keywords = list(set(post.seo_keywords + hashtags))[:15]
            post.twitter_hashtags = " ".join(
                f"#{h.replace(' ', '').replace('-', '')}" for h in hashtags)

            bundle_tweet = bundle.get("tweet_text", "").strip()
            if bundle_tweet:
                candidate = f"{bundle_tweet}\n\n{post.twitter_hashtags}"
                post.prewritten_tweet = candidate if len(
                    candidate) <= 280 else bundle_tweet[:277] + "..."
                print(
                    f"Bundle tweet attached ({len(post.prewritten_tweet)} chars)")
            else:
                post.prewritten_tweet = ""
                print("Note: no tweet_text in bundle — template fallback will be used.")

            return post

        except Exception as e:
            print(
                f"All API providers exhausted ({e}). Using local template content.")
            return self._generate_fallback_post(topic)

    # ─────────────────────────────────────────────────────────────
    # SINGLE BUNDLE CALL
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
            "deep_dive":       "Title should promise a real technical revelation, not just a topic name.",
            "tutorial":        "Title should describe the outcome, e.g. 'Build X with Y in Z minutes'.",
            "opinion":         "Title should take a stance — controversial is fine if honest.",
            "comparison":      "Title should name both options being compared.",
            "case_study":      "Title should hint at the result, e.g. 'How we cut X by 40%'.",
            "explainer":       "Title should address the confusion, e.g. 'X finally explained'.",
            "listicle":        "Title should name the number of items and promise a ranked or curated list.",
            "troubleshooting": "Title should name the specific error or symptom.",
        }.get(format_name, "Write a specific, benefit-driven title.")

        messages = [
            {
                "role": "system",
                "content": (
                    f"{author_note}\n\n"
                    "Write with a specific, personal voice. Use 'I' and 'we' where natural. "
                    "Avoid: 'in today's fast-paced world', 'it is important to note', 'crucial aspect', "
                    "'dive into', 'delve into', 'In conclusion', 'leverage', 'unleash', 'game-changer'. "
                    "Every paragraph must make a specific claim — no filler. "
                    "Be willing to say 'I got this wrong at first' or 'this surprised me'. "
                    "Name actual tools, actual version numbers, actual failure scenarios. "
                    f"Format type for this post: {format_name.upper()}. {format_note}\n\n"
                    "IMPORTANT: Respond with ONLY a valid JSON object — no markdown fences, "
                    "no preamble, no trailing commentary."
                ),
            },
            {
                "role": "user",
                "content": f"""Write a 2500-word {format_name} blog post about: "{topic}"{keyword_text}

{title_guidance}{existing_hint}

Respond with ONLY a JSON object in this exact shape:
{{
  "title": "<specific title under 60 chars>",
  "content": "<full markdown article — no title heading at top>",
  "meta_description": "<under 155 chars — must include all three: \
(1) a specific number or outcome e.g. 'Cut latency by 40%' or 'under 10ms response time', \
(2) the primary keyword from the title, \
(3) an implied reader benefit. \
Bad example: 'A guide to Redis caching.' \
Good example: 'Cut API response time by 60% with Redis caching — connection pooling, eviction policies, and the mistakes that cause cache stampedes.' \
Never start with 'This post', 'In this article', 'A guide to', 'Learn about', 'We will', or 'You will learn'.>",
  "tweet_text": "<a single X/Twitter post, max 240 chars (hashtags will be appended separately). \
Rules: \
(1) Open with tension, a specific number, or an honest admission — never the topic name alone or 'Just published'. \
(2) Hook line under 10 words. \
(3) 2-3 lines of complete, natural prose — no truncated sentences, no trailing ellipsis mid-thought. \
(4) End with a short action cue on its own line e.g. 'Full breakdown 👇' or 'Here is why 👇'. \
(5) Write like a developer talking at a meetup, not a content scheduler. \
(6) Close with a reply-inviting question on a new line e.g. 'What broke first when you tried this?'. \
BAD: 'Everyone says Event-Driven Architecture Scale is hard. The actual hard part is something else. Stop event-driven… 👇' \
GOOD: 'Everyone says scaling event-driven systems is hard. It is — but not for the reason you think. The real bottleneck shows up somewhere nobody looks. Stop optimising the wrong layer 👇\n\nWhere did your event-driven setup break first?'>",
  "seo_keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"]
}}

Use EXACTLY these ## headings inside "content" (in order):
{heading_block}

Hard requirements for "content":
- Minimum 2000 words
- At least 2 code examples with language tags (```python, ```javascript, etc.)
- At least 3 concrete numbers (benchmarks, latency figures, percentages, cost figures)
- At least 1 first-person observation: something that surprised you, a mistake you made, or a result you measured
- Each section minimum 200 words
- Do NOT include the title as a # heading at the top
- The final section must end with a specific, actionable next step — not a generic "start today"
- Include a "## Frequently Asked Questions" section near the end with 3–4 questions
  written as real search queries. Answer each in 3–5 sentences.
- At least one comparison table using markdown table syntax (| col | col |)
- Each major section should have a 1–2 sentence summary at the end

Requirements for "seo_keywords": 8 items — 2 short-tail, 4 long-tail, 2 question-based.

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
            lambda t: json.loads(_sanitize(
                re.search(r'\{.*\}', t, re.DOTALL).group()
            )) if re.search(r'\{.*\}', t, re.DOTALL) else (_ for _ in ()).throw(ValueError()),
            lambda t: json.loads(_sanitize(_repair(t))),
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
            f"Model did not return valid JSON.\nRaw (first 400):\n{raw[:400]}")

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
                    "No generic padding — every sentence must add specific value."
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
                    "Return the complete article (original + new sections). "
                    "Do not repeat the title. Keep the same author voice throughout."
                ),
            },
        ]
        return await self._call_api_with_fallback(messages, max_tokens=6500)

    # ─────────────────────────────────────────────────────────────
    # LOCAL FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _generate_fallback_post(self, topic: str) -> BlogPost:
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

## Real-World Performance Numbers

| Traffic Level | Without Tuning | With Tuning | Key Difference |
|---|---|---|---|
| Under 1,000 req/min | Baseline | Baseline | Default config works |
| 1,000–50,000 req/min | +20–35% latency | +2–5% latency | Connection pooling |
| 50,000+ req/min | Bottlenecked | +40% throughput | Clustered setup |

## Common Mistakes and How to Avoid Them

No timeout, binary error handling, no pool monitoring, happy-path-only testing, and DNS caching in containers are the five mistakes I see most often. Each is fixable in under an hour once you know to look for it.

## Frequently Asked Questions

**What is {topic} and why does it matter?**
{topic} is a core concept in modern software development that directly affects reliability, performance, and maintainability. Most developers encounter it indirectly — through a slow query, a timeout, or a cascade failure — before they understand the root cause.

**How long does it take to implement {topic} correctly?**
A basic implementation takes a few hours. A production-ready one — with monitoring, error handling, and load testing — typically takes 1–2 days.

**What are the most common mistakes when using {topic}?**
The three most common mistakes are: skipping timeout configuration, treating all errors the same way, and not instrumenting the integration before going live.

**When should I NOT use {topic}?**
If your traffic is low and predictable (under a few hundred requests per minute), the operational overhead may not be worth it. Start simpler, measure, and add complexity only when your metrics demand it.

## What to Do Next

Add explicit timeouts today. Set up latency histograms this week. Run a chaos test against staging this month."""

        meta_description = (
            f"Cut {topic} failures by understanding connection pooling and retry logic — "
            f"real benchmarks, common mistakes, and a step-by-step implementation guide."
        )
        if len(meta_description) > 155:
            meta_description = meta_description[:152] + "…"

        fallback_hashtags = _derive_hashtags_from_keywords(
            [topic_lower, f"{topic_lower} tutorial",
                f"{topic_lower} best practices"],
            topic=topic, title=title, max_hashtags=5,
        )

        post = BlogPost(
            title=title, content=content, slug=slug,
            tags=[topic_lower.replace(
                ' ', '-'), 'development', 'technical-guide', 'best-practices'] + fallback_hashtags,
            meta_description=meta_description,
            featured_image=f"/static/images/{slug}.jpg",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            seo_keywords=[
                topic_lower, f"{topic_lower} tutorial",
                f"{topic_lower} best practices",
                f"how to use {topic_lower}", f"{topic_lower} performance",
            ],
            affiliate_links=[],
            monetization_data={},
        )

        post.monetization_data["used_fallback"] = True
        post.twitter_hashtags = " ".join(f"#{h}" for h in fallback_hashtags)
        post.prewritten_tweet = ""

        enhanced_content, affiliate_links = self.monetization.inject_affiliate_links(
            post.content, topic)
        post.content = enhanced_content
        post.affiliate_links = affiliate_links
        post.monetization_data.update(
            self.monetization.generate_ad_slots(enhanced_content))
        post.monetization_data["used_fallback"] = True
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
        reading_time = max(1, round(word_count / 200))

        if word_count < MIN_WORD_COUNT:
            print(
                f"Warning: saving post with only {word_count} words (min recommended: {MIN_WORD_COUNT})")

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

        print(
            f"Saved post: {post.title} ({post.slug}) — {word_count} words / ~{reading_time} min read")
        if post.affiliate_links:
            print(f"  - {len(post.affiliate_links)} affiliate links added")
        print(
            f"  - {post.monetization_data.get('ad_slots', 0)} ad slots configured")
        if hasattr(post, 'twitter_hashtags') and post.twitter_hashtags:
            print(f"  - Twitter hashtags: {post.twitter_hashtags}")
        if hasattr(post, 'prewritten_tweet') and post.prewritten_tweet:
            print(f"  - Prewritten tweet: {len(post.prewritten_tweet)} chars")
        print(
            f"  - has_code={post_data['has_code']} | has_table={post_data['has_table']}")


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
            # ── ORIGINAL TOPICS (keep all — 15% unused remain valid) ──────────

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

            # ── NEW: AGENTIC AI & CODING AGENTS ──────────────────────────────
            # Sourced from: Anthropic 2026 Agentic Coding Report, DEV.to Feb 2026,
            # EveryDev.ai Mar 2026 — highest reader interest category right now.

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

            # ── NEW: PLATFORM ENGINEERING & DEVSECOPS ────────────────────────
            # Sourced from: intelegain.com Mar 2026, checkmarx.com Mar 2026,
            # SaM Solutions Feb 2026 — fast-growing hiring category.

            "Platform engineering explained: why your team needs a paved road",
            "DevSecOps in practice: shifting security left without slowing down deploys",
            "How to build an internal developer platform on a startup budget",
            "AI-generated code and security: the new vulnerabilities teams are missing",
            "Supply chain attacks in 2026: how to protect your Python and npm dependencies",
            "Zero-trust architecture for small teams: what actually makes sense to implement",

            # ── NEW: AI TRUST, PRODUCTIVITY & CAREER IMPACT ──────────────────
            # Sourced from: Pragmatic Engineer May 2026, OfferZen 2026 salary report,
            # EveryDev.ai Mar 2026 — high engagement / debate topics.

            "AI is a force multiplier: why it makes senior devs more powerful and juniors more exposed",
            "How AI tools are changing what 'senior developer' actually means in 2026",
            "The AI productivity tax: when your tools create more work than they save",
            "Measuring the real impact of AI on your team's velocity (not just vibes)",
            "AI skills that actually affect your salary in 2026: a data-backed breakdown",
            "What happens to junior developers in a world of AI-assisted code generation",

            # ── NEW: AFRICA & GLOBAL SOUTH TECH CAREERS ──────────────────────
            # Sourced from: OnlineJobsKenya Feb 2026, Skillworks.ng Feb 2026,
            # Remote4Africa Nov 2025, Hirezar May 2026 — core audience context.

            "How developers in Nairobi and Lagos are landing $4k/month remote roles",
            "Andela, Toptal, and Arc: which platform actually works for African developers in 2026",
            "Building a portfolio that gets you hired remotely from Africa",
            "M-Pesa, Flutterwave, Paystack: integrating African payment APIs without the headaches",
            "The timezone advantage: why European startups are hiring East African engineers",
            "From local salary to global rate: the career moves that made the difference for me",
            "How to pass technical interviews for remote roles when you're self-taught",
            "Infrastructure constraints in Africa that changed how I write backend code",

            # ── NEW: SUSTAINABLE & COST-AWARE ENGINEERING ────────────────────
            # Sourced from: intelegain.com Mar 2026 (Green Software Foundation data),
            # EveryDev.ai Mar 2026 — growing reader interest as cloud bills rise.

            "Sustainable software engineering: cutting cloud carbon without cutting performance",
            "Carbon-aware deployment: scheduling workloads when the grid is cleanest",
            "How to cut your AWS bill by 40% using Graviton and spot instances",
            "The real cost of over-engineering: when simplicity beats the fancy architecture",

            # ── NEW: REAL-TIME & EDGE COMPUTING ──────────────────────────────
            # Sourced from: SaM Solutions Feb 2026, Microsoft Jan 2026 — strong
            # search volume for real-time systems content.

            "WebSockets vs Server-Sent Events vs long polling: which one for your use case",
            "Edge functions in 2026: when Cloudflare Workers and Vercel Edge actually make sense",
            "Building real-time features without a WebSocket server: practical alternatives",
            "5G and mobile-first backends: what changes when your users are always on cellular",

            # ── NEW: LOW-CODE, NO-CODE & AI-NATIVE APPS ──────────────────────
            # Sourced from: SaM Solutions Feb 2026 ($45B market at 28% CAGR),
            # DEV.to Feb 2026 — non-traditional developer wave.

            "When to build with no-code vs write the code yourself: a decision framework",
            "Non-traditional developers shipping real products: what the AI coding wave made possible",
            "How domain experts (doctors, lawyers, scientists) are building production software in 2026",
            "MCP servers explained: what they are and why every developer should understand them",

            # ── NEW: SYSTEM DESIGN & ARCHITECTURE FOR 2026 ───────────────────
            # Evergreen high-traffic category refreshed with 2026 angles.

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
                inject_personal_intro(blog_post, topic)
                blog_system.save_post(blog_post)

                generator = StaticSiteGenerator(blog_system)
                generator.generate_site()

                print(f"\nPost '{blog_post.title}' generated successfully!")
                print(f"Twitter hashtags: {blog_post.twitter_hashtags}")

                # ── Resolve final tweet text ───────────────────────────────────
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
                    print(f"  │ {line}")
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
                        print(SEP)
                        print("❌  X / TWITTER — POST FAILED (no retry)")
                        print(SEP)
                        print(f"  Error         : {post_result.get('error')}")
                        print(SEP + "\n")

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

        else:
            print(
                "Usage: python blog_system.py [init|auto|build|cleanup|audit|purge|debug|social|test-twitter|dedup|fix-descriptions]")

    else:
        print("AI Blog System — Usage: python blog_system.py [command]")
        print("Commands: init | auto | build | cleanup | audit | purge | debug | social | test-twitter | dedup | fix-descriptions")
