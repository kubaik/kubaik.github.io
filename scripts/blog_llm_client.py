#!/usr/bin/env python3
"""
scripts/blog_llm_client.py
==========================
Standalone LLM client for the AdSense content-rewrite pipeline.

WHY THIS IS SEPARATE FROM blog_system.py
----------------------------------------
blog_system.py's _call_api_with_fallback() lives inside the BlogSystem
class, requires config.yaml to initialize, and its provider chain is
tuned for JSON bundle output (title + content + meta + keywords + tweet
text in one structured response). Content rewriting needs none of that:
it sends a plain prompt, gets plain markdown back, and only cares that
the output is complete and non-empty.

This module provides that narrower interface. It's self-contained, reads
API keys directly from environment variables, and can be imported and
called by any script without dragging in the whole BlogSystem boot.

PROVIDER CHAIN
--------------
Ordered for cost + reliability of long-form text generation:

  1. DeepSeek Flash         — cheap (~$0.15/M in), 1M context, reliable
  2. Groq (gpt-oss-120b)    — fast, good quality, free tier
  3. Gemini 2.5 Flash       — good quality, free tier
  4. OpenRouter free        — live discovery, unpredictable but available
  5. Mistral Small          — solid fallback
  6. GitHub Models (gpt-4o) — if GITHUB_TOKEN is set
  7. NVIDIA NIM (Llama 70B) — if NVIDIA_API_KEY is set
  8. Z.AI (glm-4.7-flash)   — if ZAI_API_KEY is set

Falls through to the next provider on any failure: missing key, non-200,
malformed response, empty content, or output that fails a minimal
sanity check. A provider that returns garbage is treated the same as
one that fails outright.

USAGE
-----
    from blog_llm_client import generate_text

    text = await generate_text(
        messages=[
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Rewrite..."},
        ],
        max_tokens=12000,
        temperature=0.5,
    )

    # Sync convenience wrapper for scripts that aren't async:
    from blog_llm_client import generate_text_sync
    text = generate_text_sync(messages=[...], max_tokens=12000)

CLI (for testing provider connectivity before a batch run):

    python scripts/blog_llm_client.py --check
    python scripts/blog_llm_client.py --list-providers
    python scripts/blog_llm_client.py --prompt "Reply with exactly: OK"

ENVIRONMENT VARIABLES
---------------------
Set at least ONE of the following before running:

    DEEPSEEK_API_KEY      — https://platform.deepseek.com/
    GROQ_API_KEY          — https://console.groq.com/keys
    GEMINI_API_KEY        — https://aistudio.google.com/apikey
    OPENROUTER_API_KEY    — https://openrouter.ai/keys
    MISTRAL_API_KEY       — https://console.mistral.ai/
    GITHUB_TOKEN          — https://github.com/settings/tokens (with models:read)
    NVIDIA_API_KEY        — https://build.nvidia.com/
    ZAI_API_KEY           — https://z.ai/

Providers whose key is unset are skipped silently; only the ones you've
configured get tried. There is no penalty for setting only one.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Provider constants (kept in sync with blog_system.py's model IDs so
# both chains hit the same endpoints with the same settings)
# ─────────────────────────────────────────────────────────────────────

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
_DEEPSEEK_MODEL = "deepseek-flash"

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "openai/gpt-oss-120b"

_GEMINI_MODEL = "gemini-2.5-flash"

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_FALLBACK_MODELS = [
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
]
_OPENROUTER_EXCLUDE_SUBSTRINGS = (
    "safety", "safeguard", "guard", "moderat", "-mt", "content-safety",
    "thinkingmachines",
)

_MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
_MISTRAL_MODEL = "mistral-small-latest"

_GITHUB_URL = "https://models.github.ai/inference/chat/completions"
_GITHUB_MODEL = "gpt-4o"

_NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"

_ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"
_ZAI_MODEL = "glm-4.7-flash"


# ─────────────────────────────────────────────────────────────────────
# Shared HTTP plumbing
# ─────────────────────────────────────────────────────────────────────

def _get_aiohttp():
    """
    Import aiohttp lazily so `--list-providers` and `--check` on a
    stripped environment still work even if aiohttp isn't installed.
    """
    import aiohttp  # noqa: F401  local import is intentional
    return aiohttp


def _extract_message_content(payload: dict, provider_name: str) -> str:
    """
    Pull choices[0].message.content out of a chat-completion response
    and raise if it's missing, null, or empty.

    Same guard as blog_system.py's BlogSystem._extract_message_content:
    reasoning models (DeepSeek, GLM) can return HTTP 200 with null
    `content` because their thinking tokens exhausted the budget before
    any visible output was written. Raising here means the caller treats
    it as a normal provider failure and falls through to the next one,
    rather than returning None and crashing on `.strip()` two frames
    later.
    """
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"{provider_name} returned a malformed response "
            f"(no choices[0].message.content): {e}. "
            f"Raw keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )

    if not content or not str(content).strip():
        # Some reasoning models stash the visible answer here when the
        # token budget got eaten by thinking.
        reasoning = ""
        try:
            reasoning = message.get("reasoning_content") or ""
        except Exception:
            pass
        if reasoning and str(reasoning).strip():
            return str(reasoning)

        finish_reason = None
        try:
            finish_reason = payload["choices"][0].get("finish_reason")
        except Exception:
            pass
        raise RuntimeError(
            f"{provider_name} returned empty/null message content "
            f"(finish_reason={finish_reason!r})."
        )

    return content


async def _post_openai_compat(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
) -> dict:
    """
    POST to an OpenAI-compatible chat-completions endpoint and return the
    parsed JSON body. Raises RuntimeError on non-2xx, returning the
    response body in the error message so the caller can log it.
    """
    aiohttp = _get_aiohttp()
    connector = aiohttp.TCPConnector(ssl=False)  # match the trading-bot client

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        connector=connector,
    ) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            body_text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(
                    f"HTTP {resp.status}: {body_text[:300]}"
                )
            try:
                return json.loads(body_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Non-JSON response ({e}): {body_text[:200]}"
                )


# ─────────────────────────────────────────────────────────────────────
# Provider call functions
#
# Every function has the same shape:
#     async def _call_xxx(messages: List[Dict], max_tokens: int) -> str
#
# Returns the raw assistant content as a string. Raises on failure.
# The caller does not inspect exceptions — it just moves to the next
# provider in the chain.
# ─────────────────────────────────────────────────────────────────────

async def _call_deepseek(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set")

    payload = {
        "model": _DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "stream": False,
        # Thinking mode is ON by default and its tokens count against
        # max_tokens. For a long-form rewriting call with no multi-step
        # reasoning need, that can silently burn the whole budget on
        # hidden reasoning_content before any visible content is written.
        # Disabled so the full budget goes to the article body.
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = await _post_openai_compat(_DEEPSEEK_URL, headers, payload, timeout=180)
    return _extract_message_content(body, "DeepSeek")


async def _call_groq(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set")

    payload = {
        "model": _GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = await _post_openai_compat(_GROQ_URL, headers, payload, timeout=120)
    return _extract_message_content(body, "Groq")


async def _call_gemini(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    # Gemini uses a different request shape than OpenAI-compat. Flatten
    # system + user messages into a single text prompt.
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") != "system"]
    first_user = (
        ("\n\n".join(system_parts) + "\n\n" if system_parts else "")
        + (user_parts[0] if user_parts else "")
    )
    contents = [{"role": "user", "parts": [{"text": first_user}]}]
    for extra in user_parts[1:]:
        contents.append({"role": "user", "parts": [{"text": extra}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            # Gemini 2.5 Flash bills its thinking tokens against
            # maxOutputTokens. Disabled so the full budget goes to the
            # article body rather than hidden reasoning.
            "maxOutputTokens": max_tokens,
            "temperature": 0.5,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{_GEMINI_MODEL}:generateContent?key={key}"
    )

    aiohttp = _get_aiohttp()
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=180),
        connector=connector,
    ) as session:
        async with session.post(url, json=payload) as resp:
            body_text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"Gemini HTTP {resp.status}: {body_text[:300]}")
            data = json.loads(body_text)

    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini malformed response: {e}. Body: {body_text[:300]}")

    finish = candidate.get("finishReason")
    if finish not in (None, "STOP"):
        raise RuntimeError(f"Gemini finishReason={finish} (no usable content)")

    try:
        text = candidate["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini parse error: {e}")

    if not text or not text.strip():
        raise RuntimeError("Gemini returned empty content")

    return text.strip()


async def _fetch_openrouter_free_models(timeout: int = 20) -> List[str]:
    """
    Query OpenRouter's live catalog and return currently-free text-output
    model IDs, best candidates first. Raises on any failure; the caller
    falls back to the static list.

    OpenRouter retires ':free' slugs without warning, sometimes within
    days of each other, so a hardcoded list goes stale fast. This exists
    to pick up whatever's actually free right now.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    aiohttp = _get_aiohttp()
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        connector=connector,
    ) as session:
        async with session.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"models list HTTP {resp.status}")
            payload = await resp.json()

    today = datetime.now().strftime("%Y-%m-%d")
    candidates: List[Tuple[str, int, Optional[str]]] = []

    for m in payload.get("data", []):
        model_id = m.get("id", "")
        if not model_id.endswith(":free"):
            continue
        pricing = m.get("pricing", {})
        if pricing.get("prompt") != "0" or pricing.get("completion") != "0":
            continue
        if "text" not in (m.get("architecture", {}).get("output_modalities", []) or []):
            continue
        if any(bad in model_id.lower() for bad in _OPENROUTER_EXCLUDE_SUBSTRINGS):
            continue
        expires = m.get("expiration_date")
        if expires and expires < today:
            continue
        max_completion = (m.get("top_provider", {}) or {}).get(
            "max_completion_tokens"
        ) or 0
        # Too small to return a full article body.
        if max_completion and max_completion < 4000:
            continue
        candidates.append((model_id, max_completion, expires))

    if not candidates:
        raise RuntimeError("No usable free models in OpenRouter catalog")

    candidates.sort(key=lambda c: (c[2] is not None, -c[1]))
    return [c[0] for c in candidates]


async def _call_openrouter(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    try:
        candidate_models = await _fetch_openrouter_free_models()
    except Exception:
        candidate_models = list(_OPENROUTER_FALLBACK_MODELS)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kubaik.github.io",
        "X-Title": "kubaik-content-rewriter",
    }

    last_error: Optional[Exception] = None
    for model_id in candidate_models[:3]:
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.5,
        }
        try:
            body = await _post_openai_compat(
                _OPENROUTER_URL, headers, payload, timeout=180
            )
            return _extract_message_content(body, f"OpenRouter ({model_id})")
        except Exception as e:
            last_error = e
            # Move to the next candidate model.
            continue

    raise RuntimeError(
        f"All OpenRouter free-model candidates failed. Last error: {last_error}"
    )


async def _call_mistral(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise EnvironmentError("MISTRAL_API_KEY not set")

    payload = {
        "model": _MISTRAL_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = await _post_openai_compat(_MISTRAL_URL, headers, payload, timeout=180)
    return _extract_message_content(body, "Mistral")


async def _call_github_models(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("GITHUB_TOKEN")
    if not key:
        raise EnvironmentError("GITHUB_TOKEN not set")

    payload = {
        "model": _GITHUB_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = await _post_openai_compat(_GITHUB_URL, headers, payload, timeout=180)
    return _extract_message_content(body, "GitHub Models")


async def _call_nvidia(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise EnvironmentError("NVIDIA_API_KEY not set")

    payload = {
        "model": _NVIDIA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = await _post_openai_compat(_NVIDIA_URL, headers, payload, timeout=180)
    return _extract_message_content(body, "NVIDIA NIM")


async def _call_zai(messages: List[Dict], max_tokens: int) -> str:
    key = os.getenv("ZAI_API_KEY")
    if not key:
        raise EnvironmentError("ZAI_API_KEY not set")

    payload = {
        "model": _ZAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = await _post_openai_compat(_ZAI_URL, headers, payload, timeout=180)
    return _extract_message_content(body, "Z.AI")


# ─────────────────────────────────────────────────────────────────────
# Provider registry
# ─────────────────────────────────────────────────────────────────────

# Ordered best-first. Only providers whose key is set will actually be
# tried; the others short-circuit with EnvironmentError and the chain
# moves on.
_PROVIDER_REGISTRY: List[Tuple[str, str, "callable"]] = [
    ("DeepSeek",        "DEEPSEEK_API_KEY",   _call_deepseek),
    ("Groq",            "GROQ_API_KEY",       _call_groq),
    ("Gemini",          "GEMINI_API_KEY",     _call_gemini),
    ("OpenRouter",      "OPENROUTER_API_KEY", _call_openrouter),
    ("Mistral",         "MISTRAL_API_KEY",    _call_mistral),
    ("GitHub Models",   "GITHUB_TOKEN",       _call_github_models),
    ("NVIDIA NIM",      "NVIDIA_API_KEY",     _call_nvidia),
    ("Z.AI",            "ZAI_API_KEY",        _call_zai),
]


def configured_providers() -> List[str]:
    """Return the names of providers whose API keys are set in the env."""
    return [name for name, env_key, _fn in _PROVIDER_REGISTRY if os.getenv(env_key)]


# ─────────────────────────────────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────────────────────────────────

def _normalize_messages(
    messages: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
) -> List[Dict]:
    """
    Accept either a messages list or a bare prompt string and return a
    canonical messages list. Exactly one of the two arguments must be
    provided.
    """
    if messages and prompt:
        raise ValueError("Pass either `messages` or `prompt`, not both.")
    if messages:
        if not isinstance(messages, list) or not messages:
            raise ValueError("`messages` must be a non-empty list of dicts.")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(
                    "Each message must be a dict with 'role' and 'content' keys."
                )
        return messages
    if prompt:
        return [{"role": "user", "content": prompt}]
    raise ValueError("Provide either `messages` or `prompt`.")


async def generate_text(
    messages: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
    max_tokens: int = 12000,
    verbose: bool = True,
) -> str:
    """
    Try each configured provider in order. Return the first non-empty
    text response that passes a minimal sanity check.

    Raises RuntimeError if every configured provider fails. Callers
    should catch that and treat it as "this rewrite attempt failed,
    keep the original post" — the adsense_improve.py batch loop does
    exactly that.

    `verbose=True` prints one line per provider attempt. Set to False
    when calling in a tight loop.
    """
    canonical = _normalize_messages(messages=messages, prompt=prompt)

    providers = [
        (name, fn) for name, env_key, fn in _PROVIDER_REGISTRY
        if os.getenv(env_key)
    ]

    if not providers:
        raise RuntimeError(
            "No LLM providers configured. Set at least one of: "
            "DEEPSEEK_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, "
            "OPENROUTER_API_KEY, MISTRAL_API_KEY, GITHUB_TOKEN, "
            "NVIDIA_API_KEY, ZAI_API_KEY."
        )

    last_error: Optional[Exception] = None
    for name, caller in providers:
        try:
            if verbose:
                print(f"  → {name} ...", end=" ", flush=True)
            text = await caller(canonical, max_tokens)
            if not text or not text.strip():
                raise RuntimeError(f"{name} returned empty text")
            if verbose:
                print(f"OK ({len(text.split())} words)")
            return text
        except Exception as e:
            last_error = e
            if verbose:
                print(f"FAIL ({e})")
            continue

    raise RuntimeError(
        f"All configured LLM providers failed. Last error: {last_error}"
    )


def generate_text_sync(
    messages: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
    max_tokens: int = 12000,
    verbose: bool = True,
) -> str:
    """
    Synchronous wrapper around generate_text() for scripts that aren't
    async. Uses asyncio.run() internally, so don't call this from inside
    an already-running event loop.
    """
    return asyncio.run(
        generate_text(
            messages=messages,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=verbose,
        )
    )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

async def _cmd_check() -> int:
    """
    Send a tiny prompt to each configured provider and report which ones
    work. Cheap enough to run before a batch (a few tokens per provider).
    """
    configured = configured_providers()
    if not configured:
        print("No providers configured. Set at least one of the environment")
        print("variables listed in the module docstring.")
        return 1

    print(f"Checking {len(configured)} configured provider(s)...\n")

    ok = 0
    fail = 0
    for name, env_key, _fn in _PROVIDER_REGISTRY:
        if not os.getenv(env_key):
            continue
        try:
            text = await generate_text(
                prompt='Reply with exactly the word OK and nothing else.',
                max_tokens=32,
                verbose=False,
            )
            # Only try this one provider in isolation.
            # (generate_text would try the whole chain; we want a
            # per-provider verdict. Re-run by hand below.)
        except Exception:
            pass

        # Per-provider isolated check.
        try:
            caller = dict((n, f) for n, _k, f in _PROVIDER_REGISTRY)[name]
            text = await caller([{"role": "user", "content": "Reply with OK."}], 32)
            preview = text.strip().splitlines()[0][:40] if text.strip() else ""
            print(f"  ✅ {name:15s} {preview!r}")
            ok += 1
        except Exception as e:
            print(f"  ❌ {name:15s} {type(e).__name__}: {str(e)[:120]}")
            fail += 1

    print(f"\n{ok} working, {fail} failing.")
    return 0 if ok > 0 else 1


def _cmd_list_providers() -> int:
    print("Provider chain (in order):\n")
    for i, (name, env_key, _fn) in enumerate(_PROVIDER_REGISTRY, 1):
        status = "✅ configured" if os.getenv(env_key) else "❌ missing"
        print(f"  {i}. {name:15s}  [{env_key}]  {status}")
    return 0


async def _cmd_prompt(prompt: str, max_tokens: int) -> int:
    try:
        text = await generate_text(prompt=prompt, max_tokens=max_tokens)
    except Exception as e:
        print(f"\n❌ {e}")
        return 1
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone LLM client for the AdSense rewrite pipeline."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Send a tiny prompt to each configured provider and report status.",
    )
    parser.add_argument(
        "--list-providers", action="store_true",
        help="Show the provider chain and which keys are configured.",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Send a one-off prompt through the chain and print the response.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Max tokens for --prompt (default: 200).",
    )
    args = parser.parse_args()

    if args.list_providers:
        return _cmd_list_providers()
    if args.check:
        return asyncio.run(_cmd_check())
    if args.prompt:
        return asyncio.run(_cmd_prompt(args.prompt, args.max_tokens))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())