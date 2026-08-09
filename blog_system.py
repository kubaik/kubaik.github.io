import os
import json
import math
import random
import re
import yaml
import asyncio
import aiohttp
import requests
import hashlib
import subprocess
import sys

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
from utils import dedup_similarity

from adsense_fixes.internal_linker import build_posts_index, inject_internal_links

from velocity_controller import VelocityController
from adsense_fixes.link_validator import validate_post_links
from adsense_fixes.similarity_guard import SimilarityGuard
from adsense_fixes.image_optimizer import inject_alt_text, generate_og_card
from adsense_fixes.canonical_guard import validate_canonical, audit_duplicate_slugs
from adsense_fixes.schema_validator import extract_and_build_faq_schema
from adsense_fixes.content_freshness import inject_freshness_footer, get_publishing_schedule_status

try:
    from title_validator import (
        generate_display_title,
        validate_title,
        is_truncated as _title_is_truncated,
        MAX_DISPLAY_TITLE as _VALIDATOR_MAX_DISPLAY_TITLE,
    )
except ImportError:
    # Fall back to the conventional scripts/ location if it isn't importable
    # from the project root.
    from scripts.title_validator import (
        generate_display_title,
        validate_title,
        is_truncated as _title_is_truncated,
        MAX_DISPLAY_TITLE as _VALIDATOR_MAX_DISPLAY_TITLE,
    )
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

# How many times the CLI "auto" flow will pick a fresh topic and regenerate
# a whole post from scratch after a post-generation duplicate-content block
# (SimilarityGuard or the save_post()-time ContentDuplicateGate). This is
# separate from MAX_GENERATION_ATTEMPTS, which only covers retries *within*
# a single generate_blog_post() call (bundle failures / short content for
# one topic) — this constant covers "the finished article for topic A was
# blocked as a duplicate, so throw it away and generate topic B instead."
MAX_DUPLICATE_REGENERATION_ATTEMPTS = 3

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

    _VERSION_TOKEN_RE = re.compile(r'^v?\d+(\.\d+)*$', re.IGNORECASE)

    for tag in tags:
        tag = tag.lstrip('#').strip()
        if not tag:
            continue

        # PATCH-3 FIX: split on '.' too, not just whitespace/hyphen/underscore/
        # slash. Without this, a keyword like "Checkov 3.2.117" produced the
        # single fused word "3.2.117", which CamelCased into the literal tag
        # "Checkov3.2.117" — a dotted, non-slug-safe string that still becomes
        # a real crawlable /tag/checkov3.2.117/ archive page with exactly one
        # post in it. Splitting on '.' surfaces the version number as its own
        # token so it can be dropped below instead of silently baked in.
        words = [w for w in re.split(r'[\s\-_/.]+', tag) if w]

        # PATCH-3 FIX: drop pure version/numeric tokens ("3.2.117", "20",
        # "v1") entirely. A specific tool version is useful inline in the
        # article body, but as a standalone tag it only ever matches one
        # post — a thin, single-item archive page that dilutes crawl budget
        # and has zero chance of ranking. Keep the tool name, lose the
        # version suffix: "Checkov 3.2.117" -> "Checkov", not "Checkov32117".
        words = [w for w in words if not _VERSION_TOKEN_RE.match(w)]
        if not words:
            continue

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


def _truncate_description(desc: str, max_len: int = 155) -> str:
    """Shorten an existing, already-good meta_description to max_len chars
    without cutting mid-word. Unlike _derive_description, this never
    regenerates from the post body -- it's for the common case where the
    model's own description is fine, just longer than the SERP snippet
    length Google typically renders (~155-160 chars), so trimming beats
    throwing it away and deriving something generic instead."""
    desc = desc.strip()
    if len(desc) <= max_len:
        return desc

    sentences = re.split(r'(?<=[.!?])\s+', desc)
    built = ""
    for sentence in sentences:
        candidate = (built + " " + sentence).strip() if built else sentence
        if len(candidate) <= max_len:
            built = candidate
        else:
            break
    if len(built) >= 40:
        return built

    # No single sentence fit cleanly -- hard word-boundary cut + ellipsis.
    budget = max_len - 1
    truncated = desc[:budget]
    last_space = truncated.rfind(" ")
    if last_space > 40:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:-") + "…"


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
    """
    Automated quality gate tuned for AdSense / Helpful Content readiness
    at high publishing volume. Hard failures discard the post before save.
    All checks remain fully automatic — no manual review required.
    """
    warnings = []
    hard_failures = []
    word_count = len(content.split())
    lower = content.lower()

    # ── Hard failures (post is discarded) ────────────────────────────────────

    if word_count < 1800:
        hard_failures.append(
            f"Word count {word_count} is below the absolute minimum of 1800. "
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

    # Strong AI-filler phrases become hard failures (scaled content signal)
    critical_filler = [
        "in today's rapidly evolving",
        "in the ever-changing landscape",
        "in today's fast-paced",
        "in the ever-evolving",
        "as an ai language model",
        "as a large language model",
        "i cannot provide",
        "i don't have access",
        "harness the power of",
        "unlock the potential of",
        "paradigm shift",
        "game-changer",
        "revolutionize",
        "transformative",
        "state-of-the-art",
        "cutting-edge technology",
    ]
    for phrase in critical_filler:
        if phrase in lower:
            hard_failures.append(f"Critical AI-filler phrase: '{phrase}'")

    # Require concrete production signals (AdSense quality signal)
    has_versioned_tool = bool(re.search(
        r'\b(python\s*3\.\d+|node\.?js\s*(?:1[8-9]|2[0-2])|postgres(?:ql)?\s*1[4-6]|'
        r'redis\s*[67]|kubernetes\s*1\.\d+|aws\s+lambda|fastapi\s*0\.\d+|'
        r'docker\s*(?:2[0-9]|compose)|kafka\s*3\.\d+)\b',
        content, re.I))
    has_metric = bool(re.search(
        r'\b(\d+%|\d+\s*ms|\d+\s*rps|\d+[kKmM]?\s*(?:req|request|call|token)s?\b|'
        r'\$\d+|\d+\s*(?:hour|day|week)s?\s*(?:of|to)\s*(?:downtime|latency|cost)|'
        r'p\d{2}|\d+,\d{3})\b',
        content))
    has_code = content.count('```') >= 2

    if not has_versioned_tool:
        hard_failures.append(
            "Missing version-pinned tool or library reference "
            "(e.g. Python 3.12, Redis 7.2, Kubernetes 1.29). "
            "Required for AdSense-quality technical content."
        )
    if not has_metric:
        hard_failures.append(
            "Missing concrete metric or number "
            "(%, ms, rps, cost, latency, throughput). "
            "Required signal of original technical substance."
        )
    if not has_code:
        hard_failures.append(
            "Fewer than two fenced code blocks. "
            "Technical posts without code are low-value for AdSense review."
        )

    # ── Warnings (logged; post still publishes) ───────────────────────────────

    if word_count < 2200:
        warnings.append(
            f"Word count low: {word_count} (preferred target ≥ 2200)")

    title_words = set(re.sub(r'[^\w\s]', '', title.lower()).split())
    title_words -= {'the', 'a', 'an'}
    if title_words and word_count < 2000:
        first_para_words = set(
            re.sub(r'[^\w\s]', '', content[:500].lower()).split()
        )
        title_overlap = len(title_words & first_para_words) / \
            max(len(title_words), 1)
        if title_overlap > 0.95:
            warnings.append(
                f"Opening section may be a near-verbatim restatement of the title "
                f"({title_overlap:.0%} title word overlap in first 500 chars)."
            )

    concrete_marker_re = re.compile(
        r"\b(for example|a common (pattern|trap|mistake|failure)|typically|"
        r"in practice|this usually|often shows up|a known issue|documented "
        r"behavior)\b",
        re.IGNORECASE,
    )
    if not concrete_marker_re.search(content):
        warnings.append(
            "No concrete illustrative examples found. Prefer specific, "
            "well-documented patterns over pure abstraction."
        )

    if "frequently asked questions" not in lower and "## faq" not in lower:
        warnings.append(
            "No FAQ section found. FAQ structured data improves AdSense signals."
        )

    if "|" not in content:
        warnings.append(
            "No markdown table found. A comparison table signals substantive content."
        )

    if "### About this article" not in content:
        warnings.append(
            "E-E-A-T author footer missing. Run inject_eeat_signals() before saving."
        )

    milder_filler = [
        "dive into", "delve into", "it's important to note", "needless to say",
        "comprehensive guide", "this article will", "we will explore",
        "in conclusion", "let's explore", "let's dive", "look no further",
        "in this blog post", "stay tuned",
    ]
    detected = [p for p in milder_filler if p in lower]
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

# NOTE ON VOICE (read before editing this pool):
# These personas used to instruct the model to invent specific autobiographical
# incidents ("recall a real incident", "mention a library version that bit
# you", "you've actually cut bills", "not in tutorials, but in real
# codebases"). An LLM has no actual incidents to recall — those instructions
# reliably produced confident, specific-sounding claims ("we cut our AI
# inference bill 68%", "3 days after cascade failure") that are simply made
# up. That's a direct E-E-A-T and AdSense content-quality risk: it's not
# thin content, it's fabricated experience presented as fact.
#
# Fix applied: personas below keep the regional context, editorial stance,
# and stylistic voice (including genuine first-person opinion — "I think X
# is overrated" is fine, that's an opinion, not a fabricated event) but no
# longer instruct the model to invent specific incidents, specific bugs
# "that bit you", or specific savings numbers as if personally verified.
# Concrete numbers/examples are still required elsewhere in the prompt —
# they should be framed as illustrative/typical ("a common pattern is...",
# "teams in this situation often see...") rather than as a personal,
# unverifiable claim of lived experience.

_AUTHOR_CONTEXTS = [
    (
        "You are Kubai Kevin, a software engineer in Nairobi with a background in "
        "production Python and Node.js backends in fintech. Write with the specificity "
        "of someone who knows this space well — name real AWS services, real library "
        "versions, and real failure modes that are documented and well-known in the "
        "community — but present them as typical/common patterns, not as a personal "
        "incident you're recalling. Never claim to work at a company you didn't, and "
        "never invent a specific personal anecdote. Write like you're explaining "
        "to a smart colleague at a Nairobi tech meetup."
    ),
    (
        "You are Kubai Kevin, a developer who closely follows GitHub issues, "
        "production postmortems, and Hacker News discussion. You're opinionated and specific. "
        "You've followed enough hype cycles to be skeptical of new tooling by default. "
        "Write with earned skepticism — praise what deserves praise, call out what's "
        "overrated — and ground opinions in publicly documented behavior, not invented "
        "personal incidents. Your audience respects directness."
    ),
    (
        "You are Kubai Kevin, writing the explainer you wish existed when this topic was "
        "confusing to you. Focus on the part that's genuinely non-obvious or commonly "
        "misunderstood — cite the actual confusing behavior (a real error message, a real "
        "gotcha in the docs) rather than a personal story about how long it took you. "
        "Acknowledge when something is genuinely hard, not just 'initially confusing'."
    ),
    (
        "You are Kubai Kevin, a developer who reviews a lot of code and sees the same "
        "categories of mistakes repeatedly across teams. This post addresses the root "
        "cause, not just the symptom. Be empathetic — most mistakes come from following "
        "outdated tutorials, not incompetence. Name the outdated pattern before showing "
        "the better one, and describe the mistake as a common pattern rather than an "
        "invented specific incident."
    ),
    (
        "You are Kubai Kevin, a remote engineer writing for distributed teams across "
        "Lagos, Berlin, Singapore, and San Francisco. "
        "'Best practices' are often region-specific: what works smoothly "
        "on a US-East server at 50ms latency hits differently on a shared VPS in West Africa. "
        "Write with that gap in mind. Name the constraint before naming the solution."
    ),
    (
        "You are Kubai Kevin, writing for freelancers and contractors serving clients in "
        "Europe, the US, and the Gulf. "
        "Your readers aren't all in Silicon Valley — some are bootstrapping on $200/month DigitalOcean "
        "droplets, others are at Series B startups with AWS enterprise agreements. "
        "When you recommend a tool, say which budget tier it actually makes sense for."
    ),
    (
        "You are Kubai Kevin, writing for engineers shipping products used heavily in Nigeria, "
        "Ghana, and East Africa — where 'good enough for Chrome on fibre' is not the bar. "
        "Cover the well-documented realities of building for mobile-data, low-bandwidth, "
        "intermittent-connection users, and for regional payment rails like M-Pesa, "
        "Flutterwave, and Paystack. Name that constraint explicitly rather than assuming "
        "a fibre connection and a recent flagship phone."
    ),
    (
        "You are Kubai Kevin, writing for developers building government and NGO tech "
        "across sub-Saharan Africa, where real constraints are common: no credit card "
        "for AWS, users on feature phones, unreliable power during deployment windows. "
        "Not every team has a devops engineer or a $10k/month cloud budget — "
        "practical alternatives to expensive tooling matter here."
    ),
    (
        "You are Kubai Kevin, a backend engineer writing for teams that serve European "
        "users, where GDPR compliance, data residency, and audit trails are non-negotiable. "
        "When a topic touches data handling, storage, or third-party integrations, factor "
        "compliance in from the start — not as an afterthought bolted on before launch. "
        "Ground claims in documented regulatory requirements, not an invented client story."
    ),
    (
        "You are Kubai Kevin, writing for startups in Southeast Asia — "
        "Indonesia, Vietnam, the Philippines — where the goal is often 'scale to millions of users "
        "before Series A'. Cover architectures that handle large traffic on lean infrastructure. "
        "When you talk about cost optimisation, use realistic, well-documented figures and "
        "frame them as typical outcomes for this kind of setup, not as your own verified savings."
    ),
    (
        "You are Kubai Kevin, writing for freelance engineers building for clients in Brazil, "
        "Colombia, and Mexico. Cover what it's genuinely like to work in a timezone that doesn't "
        "overlap neatly with a client's, to deal with payment processors that don't support a "
        "region, and to build resilient systems when managed Kubernetes isn't in the budget. "
        "Your writing is grounded in that context — real tradeoffs, not ideal-world advice."
    ),
    (
        "You are Kubai Kevin, writing for a global audience that includes beginners in Accra "
        "reading on a phone, senior engineers in London skimming for one specific insight, and "
        "students in India following along to build their first production-grade project. "
        "Write clearly enough for the beginner, specifically enough to be useful to the senior."
    ),
    (
        "You are Kubai Kevin, writing with the perspective of someone who has watched blockchain, "
        "serverless, microservices, and now AI all get oversold and then quietly normalised. "
        "Your writing cuts through the marketing language: what does this actually do, "
        "what does it actually cost, and what breaks first under real load? Back claims with "
        "documented, publicly-verifiable behavior rather than an invented personal anecdote. "
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
        "You are Kubai Kevin, writing about security for fintech and healthtech "
        "products serving users in multiple countries. Cover the well-documented, common "
        "patterns — auth bugs, insecure direct object references, secrets committed to public "
        "repos — as known industry failure modes, not as things you personally caught. "
        "When you write about any topic that touches auth, data storage, or external APIs, "
        "fold security in naturally, not as a separate 'security considerations' section "
        "that gets skimmed. Your audience is global; the attack surface is too."
    ),
    (
        "You are Kubai Kevin, a backend engineer with a genuine interest in query plans, "
        "connection pool tuning, and p99 latency. Write with the depth of someone who "
        "understands Python service profiling, Postgres index tuning, and Node.js memory "
        "diagnostics well — grounded in how these tools actually behave, not in an invented "
        "personal debugging story. "
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
        "VOICE: Write with a specific, opinionated voice. 'I' is fine for genuine "
        "opinions and analysis ('I think X is overrated', 'the more interesting "
        "question is...') — that's a real editorial stance, not a fabricated claim. "
        "Do NOT invent specific autobiographical incidents, specific personal "
        "metrics, or specific 'this happened to me' stories — an AI system has no "
        "such incidents, and presenting invented ones as real is a factual "
        "accuracy and reader-trust problem, not just a style choice. This site "
        "discloses AI-assisted authorship (see the AI content policy page); write "
        "in a way that's honest about that, not in a way that manufactures fake "
        "first-hand experience to disguise it. Write as if explaining to a smart "
        "colleague who has 3 years of experience — skip the basics they already "
        "know, but don't assume they've seen this specific edge case before.\n\n"
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
        "CONTENT QUALITY REQUIREMENTS — the post will be rejected if it lacks:\n"
        "1. At least ONE concrete, illustrative example or scenario that makes an "
        "abstract point specific — framed as a typical/common situation ('a common "
        "failure mode here is...', 'teams running into this usually see...'), NOT "
        "as an unverifiable personal claim ('I spent three days on this'). The "
        "goal is specificity, not fabricated autobiography.\n"
        "2. At least TWO code blocks with language tags\n"
        "3. At least THREE concrete numbers (ms, %, cost, line count, version number) "
        "— present these as realistic/typical figures for the scenario, not as "
        "your own personally-measured results unless the post format is explicitly "
        "a documented case study with a real, disclosed source.\n"
        "4. At least ONE tool with a specific version number "
        "(e.g. 'Python 3.11', 'Redis 7.2', 'Node 20 LTS')\n"
        "5. A comparison table using markdown table syntax\n"
        "6. A 'Frequently Asked Questions' section with 3-4 real developer questions\n"
        "7. A specific, actionable closing step the reader can do in the next 30 minutes\n\n"
        "CREDIBILITY: Name actual tools. Name actual AWS services. Name real, "
        "well-documented error messages and failure modes. Be willing to say "
        "something is hard, or that a common approach is wrong. Generic advice "
        "with no specifics is exactly what Google's quality raters flag as "
        "low-value content — but specificity should come from real, verifiable "
        "technical detail, never from an invented personal anecdote presented as fact.\n\n"
        f"FORMAT: {format_name.upper()} — {format_note}\n\n"
        "IMPORTANT: Respond with ONLY a valid JSON object — no markdown fences, "
        "no preamble, no trailing commentary."
    )


# ─────────────────────────────────────────────────────────────────
# Personal intro injection
# ─────────────────────────────────────────────────────────────────
#
# PREVIOUS DESIGN FLAW (found in production): a fixed pool of 8 full
# sentences was selected by md5(topic) % 8 and only a 1–2 word
# {keyword} was substituted. Across 793 published posts this produced
# 100+ posts sharing an IDENTICAL opening sentence verbatim — a
# textbook "scaled content abuse" signature that both search engines
# and AdSense reviewers can detect trivially (site:yourdomain.com
# "I've seen the same" returns 100+ results).
#
# FIX: build each intro from three independently-selected clause
# pools (hook / friction / promise) plus the topic keyword. With
# 10 x 10 x 10 combinations that's 1,000 distinct skeletons before
# the keyword is even substituted, and each pool slot is selected
# from a *different* hash seed so the same topic never reuses the
# same combination as another topic that happens to collide on one
# axis. This stays 100% deterministic and automated — no manual
# review, no new dependency, no API call.

_INTRO_HOOKS = [
    "The official documentation for {keyword} is good. What it doesn't cover is what happens six months into production.",
    "I spent longer than I should have on {keyword} before understanding what was actually happening.",
    "A colleague asked me about {keyword} during a code review recently, and my first answer wasn't a good one.",
    "I've hit the same {keyword} mistake in more than one production codebase over the years.",
    "Most {keyword} guides assume a clean environment and a patient timeline.",
    "The conventional advice on {keyword} is incomplete in one specific, costly way.",
    "I ran into this {keyword} problem while migrating a service under a hard deadline.",
    "After reviewing enough code that touches {keyword}, the same failure pattern keeps showing up.",
    "{keyword} looks simple until it has to survive real traffic.",
    "There's a gap between how {keyword} is taught and how it actually behaves under load.",
]

_INTRO_FRICTIONS = [
    "The tutorials all show the happy path.",
    "The edge cases only show up once real users hit the system.",
    "It works in the simple case and breaks in a specific way under load.",
    "The answers online were either wrong or skipped the part that mattered.",
    "Production gives you neither a clean environment nor a patient timeline.",
    "Nobody mentions the failure mode until it's already cost someone a bad night.",
    "The gap between the demo and the incident report is where this actually lives.",
    "Most write-ups stop exactly where the interesting part starts.",
    "The default configuration is fine right up until it isn't.",
    "It's the kind of problem that's easy to reproduce and hard to explain.",
]

_INTRO_PROMISES = [
    "This post covers what comes after the happy path.",
    "Here's what actually worked, and why.",
    "Here's the fuller picture, with the tradeoffs left in.",
    "This is the version of the write-up that includes the part that broke.",
    "Here's what I'd tell a colleague hitting this for the first time.",
    "This walks through the fix and the reasoning, not just the patch.",
    "Here's the root cause, not just the symptom.",
    "This is what I put together after working through it properly.",
]


def _select(pool: list, seed: str) -> str:
    idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(pool)
    return pool[idx]


# ─────────────────────────────────────────────────────────────────
# X/Twitter hook examples for the bundle prompt  (FIX: hook repetition)
# ─────────────────────────────────────────────────────────────────
#
# PREVIOUS DESIGN FLAW (found in production): the "tweet_text" field in
# the bundle prompt carried exactly ONE hard-coded "Good:" example —
# "Most teams burn $8k+ on AI tools before measuring ROI...". Because
# few-shot examples are the strongest signal a model follows, the LLM
# converged on that exact cost/waste framing (and the "Most teams..."
# opener specifically) across a large share of posts, which is what
# produced the repeated hook the user noticed on kubaik.github.io.
#
# FIX: rotate through a pool of hook styles the same way title shapes
# and intro sentences already rotate elsewhere in this file. One style
# is deterministically selected per topic via `_select`, so two
# different topics get two different example shapes, and the prompt
# explicitly tells the model the example is inspiration for STYLE only
# — not text to reuse or imitate line-for-line.
_TWEET_HOOK_EXAMPLES = [
    {
        "style": "cost / waste framing",
        "example": (
            "Most teams burn $8k+ on AI tools before measuring ROI.\\n\\n"
            "Most of it goes to autocomplete nobody audits.\\n\\n"
            "Here is what actually paid off 👇"
        ),
    },
    {
        "style": "before / after contrast",
        "example": (
            "Before: two engineers, two days, one timeout nobody could explain.\\n\\n"
            "After: a single config line.\\n\\n"
            "Here's what changed 👇"
        ),
    },
    {
        "style": "docs gap",
        "example": (
            "The docs for this are good. They just skip the part that pages "
            "you at 2am.\\n\\n"
            "Here's the gap nobody mentions 👇"
        ),
    },
    {
        "style": "specific number lead",
        "example": (
            "One misconfigured connection pool added 400ms to every request.\\n\\n"
            "It took a day to find and one line to fix.\\n\\n"
            "Here's how 👇"
        ),
    },
    {
        "style": "confession / mistake",
        "example": (
            "It took three failed deploys to find the real cause of this.\\n\\n"
            "The fix was smaller than the debugging session.\\n\\n"
            "Here's what finally worked 👇"
        ),
    },
    {
        "style": "unpopular opinion",
        "example": (
            "Unpopular take: most teams optimize the wrong layer first.\\n\\n"
            "Here's the one that actually moves the needle 👇"
        ),
    },
    {
        "style": "pattern across many examples",
        "example": (
            "Reviewed a dozen implementations of this pattern.\\n\\n"
            "Almost all of them hit the same wall.\\n\\n"
            "Here's the fix that held up in production 👇"
        ),
    },
    {
        "style": "open question",
        "example": (
            "How long should this actually take a team to get right?\\n\\n"
            "Longer than the docs suggest — and here's why 👇"
        ),
    },
]


def _pick_tweet_hook_example(topic: str) -> dict:
    """Deterministically pick one hook style per topic so consecutive
    posts don't converge on the same 'Most teams burn $X' framing."""
    return _select(_TWEET_HOOK_EXAMPLES, f"tweethook:{topic}")


def inject_personal_intro(post, topic: str) -> None:
    topic_lower = topic.lower()
    stop = {"how", "to", "the", "a", "an", "for", "and", "or", "vs",
            "when", "why", "what", "which", "guide", "tutorial", "tips"}
    words = [w for w in re.sub(r'[^\w\s]', '', topic_lower).split()
             if w not in stop and len(w) > 2]
    keyword = " ".join(words[:2]) if words else topic_lower

    # Independent seeds per slot so two topics that share a hook don't
    # also share a friction/promise — this is what keeps the combined
    # sentence space large instead of collapsing back to a small pool.
    hook = _select(_INTRO_HOOKS, f"hook:{topic}").format(keyword=keyword)
    friction = _select(_INTRO_FRICTIONS, f"friction:{topic}:{keyword}")
    promise = _select(_INTRO_PROMISES, f"promise:{keyword}:{topic}")

    intro = f"{hook} {friction} {promise}"

    if intro[:30] not in post.content:
        post.content = f"{intro}\n\n{post.content}"


# ─────────────────────────────────────────────────────────────────
# E-E-A-T signal injection
# ─────────────────────────────────────────────────────────────────
#
# PREVIOUS DESIGN FLAW (found in production): a single fixed footer
# was appended verbatim to every post, including the unconditional,
# unverifiable claims "Factual claims are verified against official
# documentation before publishing," "Code examples are tested
# locally," and "the author reviews and edits every article before
# it goes live." This pipeline has no human review step (by design,
# per the automation requirement), so these claims are false on every
# one of the 774 posts that carry them. Beyond the duplicate-content
# problem of identical boilerplate on ~all posts, publishing false
# editorial-process claims at scale is an AdSense/publisher-trust
# and E-E-A-T risk in its own right if ever surfaced in a review.
#
# FIX: disclose the actual process accurately. Accurate automation
# disclosure is not penalized by Google's guidance on AI-generated
# content; fabricated human-review claims are a real liability.

_EEAT_FOOTER_TEMPLATE = """

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** {review_date}
"""


def inject_eeat_signals(post, topic: str = None) -> None:
    """Inject consistent E-E-A-T + AI-disclosure footer. Fully automatic."""
    sentinel = "### About this article"
    if sentinel in post.content:
        return
    review_date = datetime.now().strftime("%B %Y")
    footer = _EEAT_FOOTER_TEMPLATE.format(review_date=review_date)
    post.content = post.content.rstrip() + "\n" + footer


# ─────────────────────────────────────────────────────────────────
# PRE-FLIGHT SIMILARITY INDEX
# ─────────────────────────────────────────────────────────────────

_PREFLIGHT_CACHE_FILE = Path(".preflight_index.json")
_PREFLIGHT_CACHE_TTL_SECONDS = 3600

# PATCH (dedup hardening round 2): PreFlightIndex (below) runs BEFORE an
# article is even written, on just a candidate topic string, using its own
# sklearn-based TF-IDF (word+bigram, sublinear TF, "english" stopwords) —
# a genuinely different vector space from dedup_similarity.py's unigram
# IDF model, because it's answering a different question (does this idea
# sound like an existing title/summary?) than ContentDuplicateGate (does
# this finished article's body substantially overlap an existing one?).
# A previous comment here claimed this threshold was kept in exact sync
# with CONTENT_DUPLICATE_THRESHOLD via a "_validate_dedup_thresholds()"
# function — that function did not actually exist anywhere in this file,
# and the two thresholds could not mean the same thing anyway since the
# vector spaces differ. Lowered from 0.60 to 0.50 for defense-in-depth
# (same "increase strictness" pass as the content gate below), but this
# is a coarse pre-filter, not a guarantee that matches the content gate
# 1:1. See _validate_dedup_thresholds() below, which now actually exists
# and only asserts internal sanity (both values are valid similarity
# thresholds in (0, 1) and the pre-flight filter isn't looser than the
# content gate), not that the two algorithms agree pairwise.
_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD = 0.50
_PREFLIGHT_MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────────
# Post-generation content quality gate (full-body near-duplicate check)
# ─────────────────────────────────────────────────────────────────
# PreFlightIndex (below) screens *topics/titles* before generation even
# starts. It cannot catch the case where two differently-worded topics
# still converge on a near-identical article body — e.g. "prompt
# injection basics" and "how to defend against prompt injection" can
# clear the title/topic check yet produce 80%+ overlapping content.
#
# This gate re-checks the *actual generated article body* against every
# already-published post immediately before publish.
#
# PATCH (dedup hardening round 2): this gate used to compute its own
# raw term-frequency vectors (NO IDF weighting, content only, no title,
# its own stopword list) — a different vector space from the audit tool
# in content_quality_scanner.py, despite a comment here previously
# claiming the two "agree on what counts as a duplicate." They didn't:
# the audit tool found published pairs at 0.52-0.73 similarity under its
# IDF+title-weighted method that this gate's plain-TF method had scored
# below its own 0.60 threshold at publish time, letting them through.
#
# Fix: this gate now imports the exact same tokenizer/IDF/cosine
# functions from dedup_similarity.py that content_quality_scanner.py
# uses, so "0.45 similarity" means the same thing in both places. The
# threshold itself is also lowered (was 0.60) as part of the same
# strictness pass, and is configurable via config.yaml's
# duplicate_similarity_threshold so it can be retuned without a code
# change if it proves too aggressive.
CONTENT_DUPLICATE_THRESHOLD = dedup_similarity.DUPLICATE_SIMILARITY_THRESHOLD


def _validate_dedup_thresholds() -> None:
    """Sanity-check the two duplicate thresholds at import time. This
    can't make the pre-flight (sklearn, bigram) and content-gate
    (dedup_similarity, unigram) vector spaces produce identical numbers
    for the same pair of posts — they're different algorithms answering
    different questions — but it can catch the specific failure mode of
    someone loosening one threshold without noticing the other, or
    setting either to a nonsensical value."""
    assert 0.0 < CONTENT_DUPLICATE_THRESHOLD < 1.0, (
        f"CONTENT_DUPLICATE_THRESHOLD={CONTENT_DUPLICATE_THRESHOLD} must be in (0, 1)"
    )
    assert 0.0 < _PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD < 1.0, (
        f"_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD={_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD} must be in (0, 1)"
    )
    # The pre-flight filter runs first and cheaper; it should never be
    # LOOSER than the full-body gate, or a post that would've been
    # blocked pre-generation could still slip through if pre-flight's
    # threshold were the higher (more permissive) number in practice.
    # (Not a proof the algorithms agree — just a guard against an
    # obviously backwards configuration.)
    assert _PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD <= CONTENT_DUPLICATE_THRESHOLD + 0.25, (
        "_PREFLIGHT_TFIDF_SIMILARITY_THRESHOLD is far looser than "
        "CONTENT_DUPLICATE_THRESHOLD — review both before shipping."
    )


_validate_dedup_thresholds()


class ContentDuplicateGate:
    """
    Full-body near-duplicate detector. Unlike PreFlightIndex (which only
    ever sees a short candidate topic string), this compares the complete
    generated article text (title + body) against the complete body of
    every already published post, so it catches overlap that only shows
    up once the article is actually written.

    Uses dedup_similarity.py's shared tokenizer/IDF/cosine so this gate
    and content_quality_scanner.py's offline audit always compute the
    same number for the same pair of posts.
    """

    def __init__(self, docs_dir: Path, threshold: float = CONTENT_DUPLICATE_THRESHOLD):
        self.docs_dir = docs_dir
        self.threshold = threshold

    def check(self, title: str, content: str, exclude_slug: str = "") -> tuple:
        """
        Returns (is_duplicate, matched_slug, matched_title, score).
        `exclude_slug` lets a post being refreshed/regenerated skip
        comparing against its own prior version.

        Raises on unexpected errors (e.g. a corrupt post.json) instead of
        swallowing them — see the call site in save_post(), which now
        treats a gate failure as "refuse to publish," not "publish
        without protection." A duplicate gate that can be crashed into
        silence isn't a duplicate gate.
        """
        CANDIDATE_KEY = "__candidate__"
        documents: Dict[str, Tuple[str, str]] = {
            CANDIDATE_KEY: (title, content)}

        if self.docs_dir.exists():
            for post_dir in self.docs_dir.iterdir():
                if not post_dir.is_dir() or post_dir.name == "static":
                    continue
                if exclude_slug and post_dir.name == exclude_slug:
                    continue
                post_json = post_dir / "post.json"
                if not post_json.exists():
                    continue
                try:
                    with open(post_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                    # A single unreadable post.json shouldn't take down
                    # duplicate protection for the whole gate — skip just
                    # this one file, loudly, and keep checking the rest.
                    print(f"  ⚠️  ContentDuplicateGate: skipping unreadable "
                          f"{post_json} ({exc})")
                    continue
                existing_content = data.get("content", "")
                if not existing_content:
                    continue
                documents[post_dir.name] = (
                    data.get("title", ""), existing_content)

        if len(documents) < 2:
            return False, "", "", 0.0

        vectors = dedup_similarity.build_corpus_vectors(documents)
        candidate_vec = vectors[CANDIDATE_KEY]

        best_score = 0.0
        best_slug = ""
        for slug, vec in vectors.items():
            if slug == CANDIDATE_KEY:
                continue
            score = dedup_similarity.cosine(candidate_vec, vec)
            if score > best_score:
                best_score = score
                best_slug = slug

        best_title = documents[best_slug][0] if best_slug else ""
        return best_score >= self.threshold, best_slug, best_title, best_score


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

        self._log_key_status()

        self.api_key = (
            self.groq_key or self.openrouter_key or self.cerebras_key
            or self.mistral_key or self.nvidia_key or self.gemini_key
            or self.github_token
        )

        self.monetization = MonetizationManager(config)
        # NOTE (found in review, 2026): this instantiation is currently
        # inert — see the STATUS note at the top of hashtag_manager.py.
        # Nothing calls self.hashtag_manager.get_daily_hashtags() or any
        # other method; live hashtags come entirely from
        # _derive_hashtags_from_keywords() further down in this file.
        # Left in place rather than removed so this diff stays additive;
        # decide (per that file's note) whether to delete this line and
        # the HashtagManager/add_hashtags_to_post import, or actually wire
        # this in.
        self.hashtag_manager = HashtagManager(config)

        self.preflight_index = PreFlightIndex(docs_dir=self.output_dir)
        self.content_duplicate_gate = ContentDuplicateGate(
            docs_dir=self.output_dir,
            threshold=config.get(
                "duplicate_similarity_threshold",
                dedup_similarity.DUPLICATE_SIMILARITY_THRESHOLD,
            ),
        )

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
                    "5. Preserve the article's overall voice and tone. If the existing text "
                    "contains a specific fabricated personal incident presented as fact "
                    "(e.g. 'I spent three days debugging...' attached to an invented "
                    "scenario), rephrase it as a general, typical pattern instead of "
                    "preserving it as a personal claim — do not invent new ones either.\n"
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
                # PATCH (dedup hardening): this used to "proceed with last
                # candidate" here, which meant a topic that had just been
                # confirmed too-similar 3 times in a row got generated and
                # published anyway, on the theory that the post-generation
                # gate would catch it. It didn't always agree (different
                # threshold), so duplicates reached publish. Fail closed
                # instead: skip this topic entirely for today's run rather
                # than force through content we already know is a near-copy.
                raise TopicExhaustedError(
                    f"Pre-flight max retries ({_PREFLIGHT_MAX_RETRIES}) reached for "
                    f"'{current_topic}' (last score {pf_score:.2f} vs '{match_title}'). "
                    "Refusing to generate a post we've already flagged as a duplicate. "
                    "Caller should select a different topic from the pool, or skip "
                    "this generation slot for today."
                )

            try:
                current_topic = await self._ask_llm_for_distinct_topic(
                    blocked_topic=current_topic,
                    similar_title=match_title,
                    similarity_score=pf_score,
                )
                current_keywords = None
                print(f"  LLM suggested: '{current_topic}'")
            except Exception as exc:
                # PATCH (dedup hardening): if the LLM can't even suggest a
                # distinct topic, that's a signal the well is basically dry
                # for this angle - don't silently fall through to generating
                # the still-blocked topic.
                raise TopicExhaustedError(
                    f"Pre-flight blocked '{current_topic}' and the LLM topic-rescue "
                    f"call failed ({exc}). Refusing to generate the blocked topic."
                ) from exc

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

                # PATCH: the old logic did a blind `title[:55]` slice-and-cut,
                # which is exactly what produced mid-sentence truncations like
                # "...the 60ms latency you" or "...AI microservices in".
                # Keep the model's full title intact, and derive a SERP-safe
                # display title using the same rules as title_validator.py
                # (respects natural break points, never ends on a weak word).
                full_title = title
                title = generate_display_title(
                    full_title, _VALIDATOR_MAX_DISPLAY_TITLE)

                _title_check = validate_title(title, full_title)
                if _title_check["errors"]:
                    print(f"  Title validation errors for '{title}':")
                    for _err in _title_check["errors"]:
                        print(f"    ❌ {_err}")
                if _title_check["warnings"]:
                    for _warn in _title_check["warnings"]:
                        print(f"    ⚠  {_warn}")

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

                _META_MAX_LEN = 155
                if len(meta_description) > _META_MAX_LEN:
                    print(f"Warning: meta_description is {len(meta_description)} chars "
                          f"(> {_META_MAX_LEN}) — trimming to fit the SERP snippet length.")
                    meta_description = _truncate_description(
                        meta_description, _META_MAX_LEN)

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
                regenerated = await self._regenerate_title(
                    title=title,
                    content=content,
                    topic=current_topic,
                    existing_titles=existing_titles_now,
                )
                full_title = regenerated.strip().strip('"')
                title = generate_display_title(
                    full_title, _VALIDATOR_MAX_DISPLAY_TITLE)
                _title_check = validate_title(title, full_title)
                if _title_check["errors"] or _title_check["warnings"]:
                    for _err in _title_check["errors"]:
                        print(f"    ❌ {_err}")
                    for _warn in _title_check["warnings"]:
                        print(f"    ⚠  {_warn}")
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

            # FIX (found in review, 2026): extract_and_build_faq_schema()
            # was imported at module load time but never actually called
            # anywhere. static_site_generator.py's _generate_article_schema()
            # already has the wiring on the *reading* side — it looks for
            # post.monetization_data.get('faq_schema', '') and appends it
            # as a second JSON-LD block whenever present — but nothing on
            # the *writing* side ever populated that key. Net effect: any
            # post whose content included a "## FAQ" or "## Frequently
            # Asked Questions" section never got FAQPage structured data,
            # even though three separate pieces of the machinery for it
            # (extraction, storage field, template rendering) all existed.
            faq_schema = extract_and_build_faq_schema(
                post.content,
                self.config.get('base_url', 'https://kubaik.github.io'),
                post.slug,
            )
            if faq_schema:
                post.monetization_data['faq_schema'] = faq_schema
                print("  ✅ FAQ schema extracted and attached to post.")

            # Preserve the complete, untruncated title alongside the
            # SERP-safe display title (see title_validator.py).
            post.full_title = full_title

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
        hook_example = _pick_tweet_hook_example(topic)

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

        hook_style = hook_example["style"]
        hook_example_text = hook_example["example"]

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

CONTENT QUALITY BAR — YOUR POST MUST SATISFY ALL OF THESE:
1. Minimum 2000 words of ORIGINAL, substantive content. No filler.
2. At least ONE concrete, illustrative scenario or example that makes an abstract
   point specific — framed as a typical/common situation ("a common trap here
   is…", "this usually shows up when…"), NOT as an invented personal claim
   ("I ran into this when…", "I spent two weeks on this…"). This site discloses
   AI-assisted authorship; specificity should come from real, verifiable
   technical detail, not from fabricated first-hand stories.
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

TITLE FORMAT — do NOT default to a "Cut/Reduce/Improve X% with Y" pattern. That
is one of at least five acceptable shapes below; if the last few posts on this
site already leaned on percentage-cut titles, deliberately pick a DIFFERENT
shape this time. A homepage full of identically-shaped titles reads as
mass-produced to both readers and search-quality systems — treat titles like
you would headlines in a real publication, where two adjacent pieces almost
never share the same construction.

Acceptable title shapes (pick whichever fits the content, not whichever is
easiest):
  - Direct claim, no number:      "Redis caching: what breaks first"
  - Named trap/mistake:            "TypeScript strict mode traps"
  - Question the reader is asking: "Why does my p99 spike after deploy?"
  - Blunt verdict/recommendation:  "Stop using cron for retries"
  - Quantified outcome (use sparingly, and vary the metric — not always a
    percentage): "Redis caching: what breaks first", "3 days lost to one
    missing depends_on", "Cut AWS costs 40%: the real levers"

TWEET HOOK FORMAT — do NOT default to a "Most teams burn $X on Y" cost/waste
opener every time. That is one of at least eight acceptable hook shapes;
this post should use the "{hook_style}" shape. If the last few posts on
this site already used a cost/waste or "Most teams..." opener, that pattern
is now over-used — pick a genuinely different sentence structure and a
different first word than recent posts. A homepage where every tweet opens
"Most teams..." reads as templated to readers, same as repeated titles.

Respond with ONLY a JSON object in this exact shape:
{{{{
  "title": "<punchy title: MAX 50 chars. No filler words (Complete/Ultimate/Guide to/Introduction to). Pick ONE of the title shapes above — do not default to the percentage-cut shape. Bad: 'A Complete Guide to Redis Caching'>",
  "content": "<full markdown article — no title heading at top>",
  "meta_description": "<under 155 chars. Must include: (1) the primary keyword, (2) an implied reader benefit, and (3) EITHER a specific number/outcome OR a specific named risk/mistake — do not make every post's meta description lead with a percentage; vary it the same way the title shape varies. Never start with 'This post', 'In this article', 'A guide to', 'Learn about', 'We will', or 'You will learn'. Good: 'Cut API response time 60% with Redis caching — connection pooling, eviction policies, and the cache stampede mistake most teams make.' Bad: 'A guide to Redis caching for developers.'>",
  "tweet_text": "<X/Twitter hook body — STRICT MAX 180 chars. NO url, NO hashtags (added automatically). Third-person voice only (they/teams/most developers). Complete sentences, no trailing ellipsis. End with an action cue like 'Full breakdown 👇' or 'Here is why 👇'. Use the '{hook_style}' shape shown below for STYLE and STRUCTURE ONLY — write your own sentences about THIS post's actual content, do not reuse or lightly reword the example's wording, numbers, or topic. Example of the '{hook_style}' shape: '{hook_example_text}'. Bad: 'I burned $8k...' (first person), 'Teams overspend on AI... realize...' (truncated), or copying the example's specific numbers/claims into an unrelated post.>",
  "seo_keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"]
}}}}

Use EXACTLY these ## headings inside "content" (in order):
{heading_block}

Hard requirements for "content":
- Minimum 2000 words
- At least 2 code examples with language tags (```python, ```javascript, etc.)
- At least 3 concrete numbers (benchmarks, latency ms, percentages, cost figures) —
  present as realistic/typical figures for the scenario, not as your own
  personally-measured results
- At least 1 concrete illustrative example: a specific, well-documented failure
  mode, error message, or gotcha, framed as a common/typical occurrence
- Each section minimum 200 words
- Do NOT include the title as a # heading at the top
- The final section must end with a specific, actionable next step — not a generic "start today"
- "## Frequently Asked Questions" section near the end with 3–4 questions written as
  real search queries (the kind a developer would type into Google).
  Answer each in 3–5 sentences.
- At least one comparison table using markdown table syntax
- OPENING: The introduction should end with one specific, concrete sentence that
  frames the real problem this post solves — e.g. "The part that trips people up
  is X, and that's what this post actually covers." Do NOT invent a specific
  personal incident or claim a first-hand experience the model doesn't have
  ("I spent three days debugging...") — ground the hook in the technical problem
  itself, not a fabricated autobiography.
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
        elif len(data["meta_description"].strip()) > 155:
            print("Note: meta_description too long from API response — trimming.")
            data["meta_description"] = _truncate_description(
                data["meta_description"].strip(), 155
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

        # ── Full-body near-duplicate gate ───────────────────────────────
        # Runs against every already-published post's actual content, not
        # just the topic/title that was checked before generation. This is
        # the last line of defense against the "8+ articles on prompt
        # injection" problem: two posts can have unrelated-sounding titles
        # and still land here with near-identical bodies.
        #
        # PATCH (dedup hardening round 2): this used to catch ANY exception
        # here, set is_dup=False, print a console warning, and let the post
        # publish anyway ("non-fatal"). That meant a single bad post.json
        # elsewhere in docs/ (bad encoding, truncated JSON, whatever) could
        # silently disable duplicate protection for every post published
        # until someone happened to notice the warning in a build log. The
        # entire point of this gate is to block near-duplicates — a gate
        # that fails open under error isn't a gate. It now fails CLOSED:
        # if the check itself can't run, the post is refused rather than
        # published unprotected. If this is causing false aborts in
        # practice (e.g. one consistently unreadable legacy post.json),
        # fix that file rather than loosening this back to fail-open.
        is_dup, dup_slug, dup_title, dup_score = self.content_duplicate_gate.check(
            title=post.title,
            content=post.content,
            exclude_slug=post.slug,
        )

        if is_dup:
            raise DuplicateContentError(
                f"Refusing to save '{post.title}': body is {dup_score:.0%} similar "
                f"to already-published post '{dup_title}' ({dup_slug}), which exceeds "
                f"the {self.content_duplicate_gate.threshold:.0%} duplicate-content "
                "threshold. Merge these articles or substantially differentiate the "
                "angle before publishing."
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
        elif len(post.meta_description.strip()) > 155:
            print(f"  meta_description was {len(post.meta_description.strip())} chars "
                  f"(> 155) — trimming before save.")
            post.meta_description = _truncate_description(
                post.meta_description.strip(), 155)

        post_dir = self.output_dir / post.slug
        post_dir.mkdir(exist_ok=True)

        post_data = post.to_dict()
        post_data['word_count'] = word_count
        post_data['reading_time_minutes'] = reading_time
        post_data['has_code'] = '```' in post.content
        post_data['has_table'] = '|' in post.content

        # Store the untruncated title separately from the SERP-safe display
        # title. Falls back to the display title for posts that predate this
        # field or were constructed without going through generate_post().
        post_data['full_title'] = getattr(post, 'full_title', post.title)

        _final_check = validate_title(post.title, post_data['full_title'])
        if not _final_check['valid']:
            print(f"  ⚠️  Title failed validation at save time: {post.title}")
            for _err in _final_check['errors']:
                print(f"      ❌ {_err}")

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


class DuplicateContentError(Exception):
    """Raised by save_post() when the generated article body is a
    near-duplicate of an already-published post (see ContentDuplicateGate)."""


class TopicExhaustedError(Exception):
    """Raised by generate_blog_post() when a topic is blocked as a
    duplicate by PreFlightIndex and retries are exhausted (or the LLM
    can't suggest a distinct alternative). Callers should catch this and
    either try a different topic from the pool or skip today's generation
    slot - never treat it as "proceed anyway"."""


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
        "linkedin": "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
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
        # ── TRENDING & EMERGING (Late 2026 / 2027) ────────────────────────────────
        "MCP in production: the hidden operational costs and security gotchas nobody talks about",
        "Why MCP won the agent-tool protocol war — and what it actually changed in how we build agents",
        "Multi-agent orchestration patterns that survive production (supervisor, swarm, debate, pipeline)",
        "The failure modes we only saw after running multi-agent systems at scale for 6 months",
        "How we built a reliable multi-agent research system without LangGraph (and why we might switch back)",
        "Agent-to-Agent (A2A) vs MCP: when to use each in 2026 production systems",
        "Context engineering for long-running agents: the patterns that actually reduce hallucinations",
        "Why most production agents still need strong human-in-the-loop boundaries in 2026",
        "LLMOps in 2026: the evaluation and monitoring stack that replaced our old RAG dashboards",
        "How newest models (Claude 4 / GPT-5 era) changed our agent architecture decisions",
        "Agentic FinOps: tracking and optimizing the real cost of autonomous AI workflows",
        "The token economics problem nobody solved yet for always-on agent teams",
        "Building evaluation harnesses for multi-agent systems that developers actually trust",
        "Memory systems for production agents: what worked and what leaked context over time",
        "How we detect and contain agent drift before it creates bad user experiences",
        "Security model for MCP servers in regulated environments (fintech case study)",
        "The governance layer we added after our first agent caused a compliance incident",
        "On-device and edge agents in 2026: when they finally beat cloud round-trips for African users",
        "How we run local LLM agents for sensitive fintech workflows without sending data abroad",
        "World models and physical AI: what they mean for backend engineers in 2026",
        "Purpose-built AI platforms vs general platforms: the decision we had to make in 2026",
        "Agentic cost management: using AI to optimize our own AI spend (and where it backfired)",

        # ── AI Engineering & LLMOps (Advanced) ────────────────────────────────────
        "How we version and rollback production agents without breaking downstream systems",
        "The hidden latency tax of multi-agent handoffs and how we reduced it by 60%",
        "Building durable agent workflows that survive restarts, model changes, and network blips",
        "Why structured logging + model pinning became non-negotiable once we had 15+ agents in production",
        "Evaluation-driven development for agents: the loop that replaced vibe testing",
        "How we measure 'agent reliability' in a way that correlates with user trust",
        "The tool-use patterns that scaled and the ones that created thundering herd problems",
        "Context window management strategies for agents that run for hours or days",
        "How we added circuit breakers and bulkheads to agent systems after the first cascade failure",
        "Productionizing 'computer use' style agents without giving them dangerous permissions",

        # ── Platform Engineering for AI Teams ─────────────────────────────────────
        "How our Internal Developer Platform evolved to support AI feature development in 2026",
        "The platform abstractions that made agent development 3x faster for our teams",
        "Why most platform teams are still building for 2024 developer workflows in an agentic world",
        "Building golden paths for AI features that don't become maintenance nightmares",
        "How we measure platform value when half the 'code' is now prompts and agent graphs",
        "The self-service AI tooling layer we built so product teams could experiment safely",

        # ── Cost, FinOps & Infrastructure for AI Workloads ────────────────────────
        "How we cut our monthly AI spend by 55% after implementing real token attribution",
        "Agentic FinOps: the dashboards and alerts that finally made AI costs visible to leadership",
        "The real cost of always-on vs on-demand agents in a Nairobi-based SaaS",
        "How we use spot instances + smart retry logic for non-critical agent workloads",
        "FinOps patterns that work when your biggest variable cost is now LLM tokens, not EC2",
        "Why traditional cloud cost tools failed us once agents started making autonomous decisions",
        "Building unit economics for AI features that product and finance teams can both understand",

        # ── Observability, Reliability & Incident Response for Agents ─────────────
        "What traditional observability missed when we introduced our first production agents",
        "How we built agent-specific tracing that actually helped during incidents",
        "The postmortems we now write differently because an agent made the wrong decision",
        "Building SLOs for agentic features that don't just measure latency and error rate",
        "How we detect when an agent is 'working' but producing low-quality or harmful output",
        "The on-call changes we made after agents started creating incidents at 2am",

        # ── Security, Governance & Compliance for AI Systems ──────────────────────
        "How we implemented least-privilege access for agents that need to call 30+ tools",
        "The prompt and tool injection attacks we actually saw in production (and how we blocked them)",
        "Building audit trails for agent decisions that satisfy both compliance and debugging needs",
        "Why agent identity and authentication became harder than we expected in 2026",
        "How we do red-teaming for internal agents without slowing down development velocity",
        "Data residency and sovereign AI constraints for African fintech using global models",

        # ── Africa & Emerging Market Specific AI Engineering ──────────────────────
        "How we built low-latency agent features for users on intermittent 3G/4G connections in East Africa",
        "The cost and reliability tradeoffs of running agents for users who pay in local mobile money",
        "Building AI features that work across M-Pesa, Paystack, and Flutterwave failure modes",
        "Why global AI best practices often fail in markets with high mobile data costs and latency",
        "How Nairobi teams are using local + cloud model routing to stay competitive on cost and speed",
        "Offline-capable agent workflows for field agents and last-mile operations in Africa",
        "The regulatory and compliance realities of deploying autonomous agents in African fintech in 2026",

        # ── Frontend, DX & Tooling in the Agent Era ───────────────────────────────
        "How Cursor, Claude Code, and Windsurf actually changed our daily engineering workflow in 2026",
        "The developer experience gaps that still exist when building and debugging multi-agent systems",
        "How we test and review agent-generated code and workflows at team scale",
        "Building internal tools that help non-AI engineers work safely with agents",

        # ── Career, Leadership & Team Dynamics in AI-Accelerated Teams ────────────
        "How the role of 'AI Engineer' evolved in our team throughout 2026",
        "The skills that became table stakes for senior engineers once agents were in production",
        "How we run code reviews and architecture decisions when significant portions of the system are agent-orchestrated",
        "Building healthy team norms around AI tool usage without creating two classes of engineers",
        "The leadership challenges of managing teams where output velocity increased dramatically but ownership became blurrier",
        "How African engineering teams are adapting hiring and onboarding for the agentic era",

        # ── Broader System Design & Architecture Trends ───────────────────────────
        "Event-driven vs agent-driven architectures: when each wins in 2026",
        "How durable execution platforms (Temporal, Inngest, etc.) changed our agent workflow thinking",
        "The database and state management patterns that survived heavy agent usage",
        "Designing systems that can gracefully degrade when an upstream agent or model is slow or wrong",
        "How we version and evolve agent capabilities without breaking existing users and integrations",

        # ── Hard Lessons & Troubleshooting ────────────────────────────────────────
        "The production incident caused by an agent that followed instructions too literally",
        "Why our beautiful multi-agent research system quietly degraded over three months",
        "The MCP server security mistake that could have exposed internal tools",
        "How we recovered after an agent made thousands of low-value API calls overnight",
        "The evaluation gap that let a subtle model behavior change slip into production",
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

            # FIX (found in review, 2026): VelocityController exists, is
            # fully documented ("HOW TO INTEGRATE... BEFORE calling
            # generate_blog_post()"), and is wired into a *manual*
            # `velocity status`/`velocity reset` CLI command — but was
            # never actually called from the one place that matters: this
            # `auto` entry point, which is what a scheduled GitHub Action
            # runs unattended. Without this check, nothing stops `auto`
            # from being triggered more times in a day than the age-based
            # cap allows (e.g. a manual re-run, a workflow_dispatch storm,
            # or a misconfigured cron), which is precisely the "10 posts
            # in 10 minutes looks like a content farm" signal this module
            # was built to prevent.
            vc = VelocityController()
            if not vc.can_publish():
                print("\n" + "═" * 68)
                print("🛑  VELOCITY LIMIT REACHED — NO POST PUBLISHED")
                print("═" * 68)
                print(f"  {vc.domain_age_summary()}")
                print(
                    "  Action : This is expected — the daily cap protects against\n"
                    "           publishing-velocity spam signals. It will reset\n"
                    "           tomorrow. Run 'python blog_system.py velocity status'\n"
                    "           for details, or set PUBLISH_DAILY_LIMIT to override."
                )
                print("═" * 68 + "\n")
                sys.exit(0)  # not a failure — clean exit, Action shows green

            blog_system = BlogSystem(config)

            try:
                topic = pick_next_topic(
                    preflight_index=blog_system.preflight_index
                )
            except Exception as e:
                print(f"Unexpected error picking a topic: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            # ── Outer duplicate-regeneration loop ───────────────────────────
            # generate_blog_post() already retries internally (across up to
            # MAX_GENERATION_ATTEMPTS topics) for bundle failures and short
            # content. But a post can also fail AFTER it's fully written and
            # post-processed — SimilarityGuard's topic-key/body check, or the
            # save_post()-time ContentDuplicateGate — because both of those
            # only have something to compare once the article exists. This
            # loop catches exactly that case: throw the blocked draft away,
            # pick a fresh topic, and generate a whole new post, up to
            # MAX_DUPLICATE_REGENERATION_ATTEMPTS times, before giving up.
            attempted_topics: List[str] = []
            guard = None
            blog_post = None

            for dup_attempt in range(1, MAX_DUPLICATE_REGENERATION_ATTEMPTS + 1):
                attempted_topics.append(topic)

                try:
                    blog_post = asyncio.run(
                        blog_system.generate_blog_post(topic))

                except TopicExhaustedError as e:
                    # Not a pipeline failure - this is the dedup gate working as
                    # intended. Exit 0 so the scheduled Action doesn't show red
                    # every time the topic pool is temporarily saturated, but
                    # print loudly so it's visible in the run log.
                    print("\n" + "═" * 68)
                    print(
                        "⏭️   NO POST PUBLISHED TODAY — every candidate topic was a duplicate")
                    print("═" * 68)
                    print(f"  Reason : {e}")
                    print(
                        "  Action : This is expected occasionally. If it happens on most\n"
                        "           runs, add fresh entries to content_topics in config.yaml -\n"
                        "           the existing pool has been substantially covered already."
                    )
                    print("═" * 68 + "\n")
                    sys.exit(0)

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

                dup_detected = False
                dup_reason = ""

                # Quality validation runs FIRST, per similarity_guard.py's own
                # "HOW TO INTEGRATE" docstring (call SimilarityGuard *after*
                # _validate_content_quality()) — hard failures are a cheap,
                # local check and should reject obviously broken content
                # before we spend time building/querying the similarity index.
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

                try:
                    guard = SimilarityGuard(docs_dir=blog_system.output_dir)
                    sim_result = guard.check(blog_post)
                    if sim_result.is_blocked:
                        dup_detected = True
                        dup_reason = f"SIMILARITY BLOCK: {sim_result.reason}"
                    else:
                        for warning in sim_result.warnings:
                            print(f"  ⚠️  Similarity: {warning}")
                except Exception as sim_err:
                    print(
                        f"  ⚠️  SimilarityGuard failed (non-fatal): {sim_err}")

                if not dup_detected:
                    inject_personal_intro(blog_post, topic)
                    inject_eeat_signals(blog_post, topic)
                    inject_freshness_footer(blog_post)

                    try:
                        injected_imgs = inject_alt_text(blog_post)
                        if injected_imgs:
                            print(
                                f"  🖼  {injected_imgs} image alt text(s) injected.")
                    except Exception as e:
                        print(
                            f"  ⚠️  Alt text injection failed (non-fatal): {e}")

                    try:
                        posts_index = build_posts_index(blog_system.output_dir)
                        base_path = config.get("base_path", "")
                        inject_internal_links(
                            blog_post, posts_index, base_path=base_path)
                    except Exception as e:
                        print(
                            f"  ⚠️  Internal link injection failed (non-fatal): {e}")

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
                        print(
                            f"  ⚠️  Canonical validation failed (non-fatal): {e}")

                    try:
                        blog_system.save_post(blog_post)
                        # Record the publish AFTER save_post() succeeds, not
                        # before — matches VelocityController's own
                        # documented contract ("to avoid counting failed
                        # attempts"). A duplicate-content rejection below
                        # must not count against today's quota.
                        vc.record_publish()
                    except DuplicateContentError as e:
                        dup_detected = True
                        dup_reason = f"DUPLICATE CONTENT: {e}"

                if not dup_detected:
                    break  # success — fall through to publishing steps below

                print("\n" + "═" * 68)
                print("🔁  DUPLICATE DETECTED — DISCARDING DRAFT AND REGENERATING")
                print("═" * 68)
                print(f"  Reason : {dup_reason}")

                if dup_attempt >= MAX_DUPLICATE_REGENERATION_ATTEMPTS:
                    print(
                        f"  Action : Exhausted {MAX_DUPLICATE_REGENERATION_ATTEMPTS} "
                        "regeneration attempt(s) across different topics. Add fresh\n"
                        "           entries to content_topics in config.yaml, or review\n"
                        "           whether SimilarityGuard's thresholds need retuning\n"
                        "           (see similarity_guard.py's audit CLI)."
                    )
                    print("═" * 68 + "\n")
                    sys.exit(1)

                existing_titles_for_retry = _load_existing_titles(
                    blog_system.output_dir)
                topic = blog_system._pick_retry_topic(
                    topic, existing_titles_for_retry, exclude=attempted_topics
                )
                print(
                    f"  Action : Trying attempt {dup_attempt + 1}/"
                    f"{MAX_DUPLICATE_REGENERATION_ATTEMPTS} with new topic: '{topic}'"
                )
                print("═" * 68 + "\n")
                # loop continues with the new topic

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
                        or len(desc) > 155
                    )
                    if needs_fix:
                        if desc and len(desc) > 155 and not any(
                                desc.lower().startswith(w) for w in _weak_openers):
                            # Good description, just too long -- trim it,
                            # don't discard it and derive something generic.
                            fixed_desc = _truncate_description(desc, 155)
                            reason = "too long"
                        else:
                            fixed_desc = _derive_description(
                                data.get("content", ""), data.get("title", ""))
                            reason = "empty" if not desc else "weak opener"
                        data["meta_description"] = fixed_desc
                        with open(post_json, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(
                            f"Fixed ({reason}): {post_dir.name} → {fixed_desc[:80]}…")
                        fixed += 1
                except Exception as e:
                    print(f"Error fixing {post_dir.name}: {e}")
            print(
                f"\nFixed {fixed} posts. Run 'python blog_system.py build' to regenerate HTML.")

        elif mode == "fix-titles":
            if not os.path.exists("config.yaml"):
                print("config.yaml not found.")
                sys.exit(1)
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            blog_system = BlogSystem(config)
            docs_dir = blog_system.output_dir
            fixed = 0
            checked = 0
            for post_dir in docs_dir.iterdir():
                if not post_dir.is_dir() or post_dir.name == "static":
                    continue
                post_json = post_dir / "post.json"
                if not post_json.exists():
                    continue
                try:
                    with open(post_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    checked += 1
                    current_title = data.get("title", "")
                    # If full_title is missing (pre-dates this field), assume
                    # the stored title IS the full title — we can't recover
                    # what was truncated away, but we can still stop it from
                    # being mangled further and flag it for manual review.
                    stored_full_title = data.get("full_title", current_title)
                    result = validate_title(current_title, stored_full_title)

                    if not data.get("full_title") and _title_is_truncated(current_title):
                        print(
                            f"  ⚠️  {post_dir.name}: title looks truncated but has "
                            f"no `full_title` on record — cannot recover the "
                            f"missing words automatically. Flagging for manual fix."
                        )

                    needs_fix = (
                        not result["valid"]
                        or current_title != result["display_title"]
                        or data.get("full_title") != result["full_title"]
                    )
                    if needs_fix:
                        data["title"] = result["display_title"]
                        data["full_title"] = result["full_title"]
                        with open(post_json, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"\n  [{post_dir.name}]")
                        print(f"    Before: {current_title}")
                        print(f"    After:  {result['display_title']}")
                        for e in result["errors"]:
                            print(f"    ❌ {e}")
                        fixed += 1
                except Exception as e:
                    print(f"Error fixing {post_dir.name}: {e}")
            print(f"\nChecked {checked} posts, fixed {fixed}.")
            print("Run 'python blog_system.py build' to regenerate HTML.")

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
                "debug|social|test-twitter|dedup|fix-descriptions|fix-titles|"
                "refresh-stale|audit-links|audit-slugs|audit-freshness|velocity|"
                "preflight-rebuild|preflight-check]"
            )

    else:
        print("AI Blog System — Usage: python blog_system.py [command]")
        print("Commands: init | auto | build | cleanup | audit | purge | debug | social | "
              "test-twitter | dedup | fix-descriptions | fix-titles | refresh-stale | "
              "audit-links | audit-slugs | audit-freshness | velocity | preflight-rebuild | "
              "preflight-check")
