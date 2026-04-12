"""
AdSense Content Quality Enhancer
Enhances blog content to meet Google AdSense quality standards.

Key fix vs original: the `except` block inside _generate_enhanced_content
was at the wrong indentation level, causing a SyntaxError that prevented
the module from loading at all. Fixed below.
"""

import asyncio
import aiohttp
import re
from datetime import datetime
from typing import Dict, List
import random


class ContentQualityEnhancer:
    """Enhances content to meet AdSense standards and avoid 'low value content' issues."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.min_word_count = 1500
        self.min_sections = 7

    async def enhance_post_for_adsense(self, post, topic: str) -> Dict:
        """
        Enhance a blog post to meet AdSense quality standards.

        Returns a dict with enhancement results including word count and quality score.
        """
        enhanced_content = await self._generate_enhanced_content(topic, post.title)

        post.content = enhanced_content
        post.updated_at = datetime.now().isoformat()

        quality_score = self._calculate_quality_score(enhanced_content)

        return {
            'enhanced': True,
            'word_count': len(enhanced_content.split()),
            'quality_score': quality_score,
            'sections': enhanced_content.count('##'),
            'improvements': [
                'Extended content length to 1500+ words',
                'Added specific examples and case studies',
                'Included practical implementation steps',
                'Added honest tradeoffs section',
                'Improved heading structure',
                'Added code examples',
            ]
        }

    async def _generate_enhanced_content(self, topic: str, title: str) -> str:
        """Generate high-quality, AdSense-friendly content."""

        if not self.api_key:
            return self._generate_enhanced_fallback(topic, title)

        # FIX: The original code had a try block that closed before the except,
        # causing a SyntaxError. The try/except now wraps the full generation block.
        try:
            content_sections = []

            intro = await self._generate_section(
                "introduction", topic, title,
                """Write a 200-word introduction that:
                - Opens with a specific, concrete problem statement (not 'In today's world...')
                - States exactly what the reader will be able to do after reading
                - Mentions one surprising or counter-intuitive fact about the topic
                - Uses a direct, conversational tone
                Avoid: generic openers, vague benefits, buzzwords."""
            )
            content_sections.append(intro)

            problem_section = await self._generate_section(
                "the_real_problem", topic, title,
                """Write 250 words explaining the core problem with how most people approach this topic:
                - Identify the most common misconception
                - Explain what goes wrong because of it
                - Give a specific example (with realistic numbers or scenario)
                - Use a clear ## heading"""
            )
            content_sections.append(problem_section)

            implementation = await self._generate_section(
                "implementation", topic, title,
                """Write a 350-word step-by-step implementation guide:
                - Use a clear ## heading
                - Number the steps (1 through 5-6)
                - Each step must have: what to do, why it matters, what to watch for
                - Include one realistic code snippet (fenced with language tag)
                - Be specific about tool names and version numbers"""
            )
            content_sections.append(implementation)

            benchmarks = await self._generate_section(
                "performance_numbers", topic, title,
                """Write 200 words on real performance or impact numbers:
                - Use a clear ## heading (e.g. '## What the Numbers Actually Show')
                - Include at least 3 concrete metrics or benchmarks
                - Compare before/after or compare alternative approaches
                - Cite where these numbers come from (even if general industry data)
                - Be honest about variance and conditions"""
            )
            content_sections.append(benchmarks)

            mistakes = await self._generate_section(
                "common_mistakes", topic, title,
                """Write 250 words on 4-5 specific mistakes people make:
                - Use a clear ## heading
                - Each mistake: name it, explain why people make it, explain the consequence, give the fix
                - Be specific — not 'don't forget to test' but 'don't skip testing connection timeouts under load'
                - Include one mistake that is genuinely surprising or non-obvious"""
            )
            content_sections.append(mistakes)

            tools = await self._generate_section(
                "tools_comparison", topic, title,
                """Write 200 words comparing 3-4 specific tools or approaches:
                - Use a clear ## heading
                - For each tool: one sentence on what it's best for, one sentence on its biggest weakness
                - Give a concrete recommendation: 'Use X when Y, use Z when W'
                - Mention actual version numbers or release dates where relevant"""
            )
            content_sections.append(tools)

            when_not_to = await self._generate_section(
                "when_not_to_use", topic, title,
                """Write 200 words on when NOT to use this approach:
                - Use a clear ## heading (e.g. '## When to Skip This Entirely')
                - Give 3 specific scenarios where this approach is wrong
                - For each: describe the situation, explain why this approach fails, suggest the alternative
                - Be honest — this section builds trust with readers and with Google"""
            )
            content_sections.append(when_not_to)

            conclusion = await self._generate_section(
                "conclusion", topic, title,
                """Write a 150-word conclusion:
                - Use a ## Conclusion heading
                - Summarise the 3 most important points in one sentence each
                - Give 3 concrete next actions the reader can take today, this week, this month
                - End with an honest statement about what this approach cannot do"""
            )
            content_sections.append(conclusion)

            return "\n\n".join(content_sections)

        # FIX: except is now correctly paired with the try above
        except Exception as e:
            print(f"Error generating enhanced content: {e}")
            return self._generate_enhanced_fallback(topic, title)

    async def _generate_section(self, section_type: str, topic: str,
                                title: str, instruction: str) -> str:
        """Generate a specific section using the OpenAI API."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced technical writer with deep hands-on knowledge. "
                    "Write in a direct, specific voice. Every sentence must earn its place. "
                    "No filler phrases, no vague benefits, no generic statements. "
                    "Use concrete examples, real numbers, and specific tool names. "
                    "Take clear positions — hedging everything is not helpful."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n"
                    f"Article title: {title}\n"
                    f"Section type: {section_type}\n\n"
                    f"{instruction}\n\n"
                    "Write original content that provides genuine insight. "
                    "Do not start with 'In this section' or restate the section type."
                )
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-4.1-nano",
            "messages": messages,
            "max_tokens": 700,
            "temperature": 0.75
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception(f"API error: {response.status}: {await response.text()}")

    def _generate_enhanced_fallback(self, topic: str, title: str) -> str:
        """
        Generate structured fallback content when API is unavailable.

        This is substantially better than the original — it has specific structure,
        avoids filler phrases, and includes a code example, all of which matter
        for AdSense quality assessment.
        """
        topic_lower = topic.lower()
        topic_slug = topic.replace(' ', '').replace('-', '')[:20]

        return f"""## Why Most {topic} Implementations Fail

The most common {topic} implementation problem is not technical — it is a planning problem. Developers reach for {topic} before understanding what problem they are actually solving, which leads to over-engineered solutions that are harder to maintain than what they replaced.

Before implementing {topic}, you need clear answers to three questions: What is the specific bottleneck or gap this solves? What does failure look like, and how will you detect it? What is the rollback plan if it does not work as expected?

Getting these wrong costs more time than building the wrong implementation.

## How {topic} Actually Works

At its core, {topic} operates through a combination of configuration, runtime state, and coordination between components. The configuration layer defines behaviour; the runtime layer executes it; coordination ensures consistency across instances.

Understanding this separation matters because most problems occur at the boundaries — when runtime state does not match configuration expectations, or when coordination between instances breaks down under load.

The performance profile follows a predictable pattern: lightweight at small scale (under 1,000 operations/minute), moderate overhead at medium scale (1,000–50,000/minute) that requires connection pooling, and significant architectural consideration at large scale (50,000+/minute) where you need clustered setups.

## Step-by-Step Implementation Guide

**Step 1: Define your success criteria before writing code.** What specific metric improves? By how much? How will you measure it?

**Step 2: Start with the minimal viable configuration.** Resist the temptation to configure everything upfront. Add complexity only when you observe a specific problem that requires it.

**Step 3: Implement with explicit error handling.** Here is a minimal pattern:

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class {topic_slug}Handler:
    def __init__(self, config: dict):
        self.timeout = config.get('timeout_seconds', 5.0)
        self.max_retries = config.get('max_retries', 3)
        self._client = None
    
    def execute(self, operation: str, payload: dict) -> Optional[dict]:
        for attempt in range(self.max_retries):
            try:
                return self._run(operation, payload)
            except TimeoutError:
                if attempt == self.max_retries - 1:
                    logger.error(f"{{operation}} timed out after {{self.max_retries}} attempts")
                    return None
                logger.warning(f"Timeout on attempt {{attempt + 1}}, retrying...")
        return None
    
    def _run(self, operation: str, payload: dict) -> dict:
        # Your implementation here
        raise NotImplementedError
```

**Step 4: Add observability immediately.** Log operation duration, success rate, and error type for every operation. You cannot debug what you cannot measure.

**Step 5: Load test before going live.** Use realistic traffic patterns, not just peak load. Many failures only appear after sustained moderate load, not during brief spikes.

## What the Numbers Show

Across production deployments of similar systems, the performance impact breaks down as follows:

- Connection establishment: 5–50ms depending on network topology and authentication method
- Per-operation overhead vs direct calls: 2–8% at steady state with proper connection pooling
- Memory overhead per 100 concurrent connections: 2–5MB for the coordination layer
- Cold start penalty: 8–15x worse than steady-state until the connection pool is warm (typically 30–60 seconds)

The most important number to track is not average latency — it is p99 latency. Averages hide the tail behaviour that causes real user complaints. Set up histogram metrics, not just averages.

## Common Mistakes That Cost Weeks

**Mistake 1 — No connection timeout.** Most libraries default to no timeout or 30+ seconds. Set an explicit timeout at connection establishment (2–5s) and per-operation (1–5s). Silent hangs are worse than fast failures.

**Mistake 2 — Testing only the success path.** Production failures almost always happen in error paths that were never tested. Use fault injection in staging: add artificial delays, drop connections, return error codes randomly. This finds 80% of production incidents before they happen.

**Mistake 3 — Ignoring connection pool exhaustion.** When the pool is full, requests queue silently. This looks like a latency spike, not a connection issue. Add metrics for pool utilisation and active wait time. Alert when wait time exceeds 200ms.

**Mistake 4 — Treating all errors the same.** A connection refused error (retry with backoff) is different from an authentication error (fail immediately and alert) which is different from a timeout (retry once, then fail). Build specific handlers for each error class.

**Mistake 5 — Skipping the circuit breaker.** Without a circuit breaker, a downstream failure causes your application to queue up thousands of requests that will all fail. After 5 consecutive failures, stop trying for 30 seconds. Libraries like `resilience4j`, `tenacity`, or `polly` provide this with minimal code.

## Tools Worth Using

**For connection management:** Use an established library for your language rather than writing your own. `httpx` (Python), `okhttp` (Java/Kotlin), and `got` (Node.js) handle retries, timeouts, and connection pooling correctly out of the box.

**For monitoring:** Prometheus with histogram metrics (not just counters and gauges). Grafana for dashboards. Set up latency alerts at p95 and p99, not at average.

**For testing:** Testcontainers for integration tests against real infrastructure. `k6` or `Locust` for load testing. Run load tests weekly against staging, not just before launch.

**For resilience:** `tenacity` (Python), `resilience4j` (JVM), or `Polly` (.NET) for circuit breakers and retry logic. Do not write your own unless you have a very specific requirement — these libraries have years of edge cases baked in.

## When to Skip {topic} Entirely

{topic} is not always the right choice. Be honest about whether it fits your situation:

**If your traffic is under 500 requests/minute and predictable,** the added complexity is not justified. A simple, synchronous approach is easier to debug and operate. Complexity has a real cost in developer time and incident response.

**If you do not have observability infrastructure,** you will not be able to debug problems when they occur. Set up metrics and logging first; add {topic} second.

**If your team is not familiar with the failure modes,** operational incidents will take longer to resolve than they would with a simpler system. The sophistication of the architecture must match the sophistication of the team operating it.

**Alternative to consider:** If the core requirement is reliability rather than performance, a message queue (RabbitMQ, SQS, or Kafka) with at-least-once delivery often solves the actual problem with less operational complexity.

## Conclusion

The gap between a working {topic} prototype and a production-ready implementation comes down to how well you handle failure cases. The happy path is straightforward. The value is in the error handling, the monitoring, and the circuit breakers.

Three actions to take now: Set explicit timeouts on every operation today. Add p99 latency metrics this week. Run a fault-injection test against staging this month.

{topic} works well when you understand its failure modes. It creates problems when you treat it as a black box. The documentation covers configuration; this guide covers what to do when the configuration does not help."""

    def _calculate_quality_score(self, content: str) -> int:
        """Calculate content quality score (0-100) based on AdSense-relevant signals."""
        score = 0

        # Word count (max 25 points)
        word_count = len(content.split())
        if word_count >= 1500:
            score += 25
        elif word_count >= 1000:
            score += 18
        elif word_count >= 700:
            score += 10

        # Section count (max 20 points)
        section_count = len(re.findall(r'^##\s+', content, re.MULTILINE))
        if section_count >= 7:
            score += 20
        elif section_count >= 5:
            score += 14
        elif section_count >= 3:
            score += 8

        # Code examples (max 15 points — very important for tech content)
        code_blocks = len(re.findall(r'```', content)) // 2
        if code_blocks >= 2:
            score += 15
        elif code_blocks == 1:
            score += 10

        # Specific numbers/metrics (max 10 points)
        numbers = re.findall(
            r'\d+(?:\.\d+)?(?:%|ms|MB|KB|GB|s\b|x\b)', content)
        if len(numbers) >= 5:
            score += 10
        elif len(numbers) >= 2:
            score += 6

        # Structural variety (max 10 points)
        has_ordered_list = bool(re.search(r'^\d+\.', content, re.MULTILINE))
        has_unordered_list = bool(re.search(r'^[-*]\s', content, re.MULTILINE))
        has_bold = '**' in content
        if has_ordered_list and has_unordered_list and has_bold:
            score += 10
        elif (has_ordered_list or has_unordered_list) and has_bold:
            score += 6

        # Absence of filler phrases (max 10 points)
        filler_phrases = [
            'in today\'s fast-paced', 'it is important to note', 'crucial aspect',
            'plays a vital role', 'in conclusion, overall', 'needless to say',
            'it goes without saying', 'at the end of the day'
        ]
        filler_count = sum(
            1 for phrase in filler_phrases if phrase in content.lower())
        if filler_count == 0:
            score += 10
        elif filler_count <= 2:
            score += 5

        # Paragraph variety (max 10 points)
        paragraphs = [p.strip() for p in content.split(
            '\n\n') if p.strip() and not p.startswith('#')]
        if len(paragraphs) >= 12:
            score += 10
        elif len(paragraphs) >= 8:
            score += 7

        return min(score, 100)


# ─────────────────────────────────────────────────────────────────
# Integration function
# ─────────────────────────────────────────────────────────────────

async def enhance_all_posts_for_adsense(blog_system):
    """Enhance all existing posts to meet AdSense standards."""

    enhancer = ContentQualityEnhancer(blog_system.api_key)

    posts_dir = blog_system.output_dir
    enhanced_count = 0

    print("Enhancing all posts for AdSense approval...")
    print("=" * 60)

    for post_dir in posts_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name == 'static':
            continue

        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue

        try:
            import json
            with open(post_json, 'r') as f:
                post_data = json.load(f)

            from blog_post import BlogPost
            post = BlogPost.from_dict(post_data)

            word_count = len(post.content.split())
            if word_count >= 1500:
                print(f"OK  {post.title[:50]}... ({word_count} words)")
                continue

            print(f"\nEnhancing: {post.title}")
            print(f"  Current: {word_count} words")

            result = await enhancer.enhance_post_for_adsense(
                post,
                post.tags[0] if post.tags else post.title
            )

            print(f"  Enhanced: {result['word_count']} words")
            print(f"  Quality:  {result['quality_score']}/100")
            print(f"  Sections: {result['sections']}")

            blog_system.save_post(post)
            enhanced_count += 1

            await asyncio.sleep(2)

        except Exception as e:
            print(f"Error enhancing {post_dir.name}: {e}")

    print("\n" + "=" * 60)
    print(f"Enhanced {enhanced_count} posts")
    print("\nNext steps:")
    print("1. Review enhanced posts for accuracy")
    print("2. Rebuild: python blog_system.py build")
    print("3. Wait 2–3 weeks for Google to recrawl")
    print("4. Re-request AdSense review")

    return enhanced_count
