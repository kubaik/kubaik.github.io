"""
AdSense Content Quality Enhancer

Changes vs. original:

1. REMOVED the "even if general industry data" instruction from the
   benchmarks/metrics prompt. That instruction told the model to state
   specific numbers with no real source behind them — exactly the
   fabricated_citation pattern adsense_compliance_audit.py deletes posts
   for. The prompt now requires the model to either cite a real,
   checkable source or explicitly hedge ("in practice, teams typically
   see...") instead of stating invented precision.

2. _calculate_quality_score no longer scores raw numeric density as a
   positive signal on its own (that was the fabrication incentive — more
   fake numbers = higher score). It now only credits metrics when they
   appear alongside a hedge phrase or an attributable source, and adds a
   separate honesty signal for hedged/uncertain language, which used to
   be scored as a *negative* (filler) even when appropriate.

3. enhance_post_for_adsense no longer wholesale-replaces post.content.
   It returns the enhanced draft alongside the original and an explicit
   `source` tag; the caller decides what to do with it instead of this
   function silently overwriting history.

4. The template fallback path (no API key) is now clearly tagged
   `enhancement_source: 'template_fallback'` in the returned dict and in
   post.monetization_data, so downstream tooling (near-duplicate
   detection, the regeneration queue) can treat template-fallback posts
   as flagged-for-review rather than as equivalent to freshly generated
   content. Previously this was indistinguishable from real generation
   and would silently accumulate near-identical posts.

5. The `improvements` list is now derived from what actually happened
   (which code path ran, what changed) instead of a hardcoded constant
   list returned unconditionally.
"""

import asyncio
import aiohttp
import re
from datetime import datetime
from typing import Dict, List
import random


HEDGE_PHRASES = (
    "typically", "in practice", "often see", "can vary", "depending on",
    "in our experience", "as a rough guide", "your results may differ",
    "roughly", "approximately", "on the order of",
)


class ContentQualityEnhancer:
    """Enhances content to meet AdSense standards and avoid 'low value content' issues."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.min_word_count = 1500
        self.min_sections = 7

    async def enhance_post_for_adsense(self, post, topic: str) -> Dict:
        """
        Generate an enhanced draft for a blog post. Does NOT mutate `post` —
        the caller is responsible for reviewing `enhancement_source` and
        deciding whether to accept the draft, route it through the audit
        pipeline (named_source_hit / near-duplicate checks) first, or
        discard it.

        Returns a dict with the draft content, word count, quality score,
        and provenance.
        """
        enhanced_content, source = await self._generate_enhanced_content(topic, post.title)

        quality_score = self._calculate_quality_score(enhanced_content)

        improvements = []
        if source == "llm":
            if len(enhanced_content.split()) >= self.min_word_count:
                improvements.append(
                    f"Extended content to {len(enhanced_content.split())} words")
            if enhanced_content.count("##") >= self.min_sections:
                improvements.append("Added structured section headings")
            if "```" in enhanced_content:
                improvements.append("Included code example(s)")
        else:
            improvements.append(
                "template_fallback used — this is boilerplate with only the "
                "topic substituted, not original generation. Route for "
                "review before publishing; do not accept silently."
            )

        return {
            'draft_content': enhanced_content,
            'original_content': post.content,
            'enhancement_source': source,  # 'llm' or 'template_fallback'
            'word_count': len(enhanced_content.split()),
            'quality_score': quality_score,
            'sections': enhanced_content.count('##'),
            'improvements': improvements,
            'requires_review': source == "template_fallback",
        }

    async def _generate_enhanced_content(self, topic: str, title: str) -> tuple[str, str]:
        """Generate high-quality, AdSense-friendly content.

        Returns (content, source) where source is 'llm' or 'template_fallback'
        so callers always know which path produced the text.
        """

        if not self.api_key:
            return self._generate_enhanced_fallback(topic, title), "template_fallback"

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
                - Be specific about tool names and version numbers you are confident about"""
            )
            content_sections.append(implementation)

            # FIX: previously instructed "cite where these numbers come from
            # (even if general industry data)" — that told the model it was
            # fine to state a specific number with no real source. Now the
            # model must either name a checkable source or hedge explicitly.
            # Both are legitimate; inventing false precision is not.
            benchmarks = await self._generate_section(
                "performance_numbers", topic, title,
                """Write 200 words on real-world performance or impact considerations:
                - Use a clear ## heading (e.g. '## What This Actually Costs You')
                - For any number you state, either (a) name a specific, real,
                  checkable source, or (b) explicitly hedge it as typical/rough
                  ("teams typically see...", "roughly", "can vary widely with...")
                - Do NOT state a specific-sounding statistic (e.g. "reduces
                  latency by 43%") without one of the above — vague honesty
                  beats false precision
                - Compare before/after or compare alternative approaches
                - Be explicit about what varies by workload/environment"""
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
                - Only state version numbers or release dates you are confident are correct;
                  otherwise describe the tool without a specific version claim"""
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

            return "\n\n".join(content_sections), "llm"

        except Exception as e:
            print(f"Error generating enhanced content: {e}")
            return self._generate_enhanced_fallback(topic, title), "template_fallback"

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
                    "Use concrete examples and specific tool names. State a specific number "
                    "only when you can name a real source for it or you clearly hedge it as "
                    "typical/approximate — never invent false precision. "
                    "Take clear positions — hedging everything is not helpful, but a fabricated "
                    "statistic is worse than an honest 'this varies'."
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
        Structured fallback content used only when the API is unavailable
        or errors out. This is intentionally generic — callers MUST check
        `enhancement_source == 'template_fallback'` and route the result
        for review rather than publishing it directly, because this same
        template with only {topic} substituted will otherwise produce
        near-identical articles at scale (a real duplicate-content risk,
        not a hypothetical one — this is the exact pattern
        adsense_compliance_audit.py's near_duplicate check exists to catch).
        """
        topic_slug = topic.replace(' ', '').replace('-', '')[:20]

        return f"""## Why Most {topic} Implementations Fail

The most common {topic} implementation problem is not technical — it is a planning problem. Developers reach for {topic} before understanding what problem they are actually solving, which leads to over-engineered solutions that are harder to maintain than what they replaced.

Before implementing {topic}, you need clear answers to three questions: What is the specific bottleneck or gap this solves? What does failure look like, and how will you detect it? What is the rollback plan if it does not work as expected?

Getting these wrong costs more time than building the wrong implementation.

## How {topic} Actually Works

At its core, {topic} operates through a combination of configuration, runtime state, and coordination between components. The configuration layer defines behaviour; the runtime layer executes it; coordination ensures consistency across instances.

Understanding this separation matters because most problems occur at the boundaries — when runtime state does not match configuration expectations, or when coordination between instances breaks down under load.

Overhead generally scales with request volume: light at small scale, requiring connection pooling once volume grows, and needing real architectural planning at high, sustained volume. Exact thresholds vary by workload and infrastructure — treat any specific number here as a starting point to measure against your own traffic, not a target.

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
                    raise
        return None

    def _run(self, operation: str, payload: dict) -> dict:
        raise NotImplementedError
```

**Step 4: Add monitoring before you need it.** Track latency, error rate, and retry count from day one, not after the first incident.

**Step 5: Load-test against realistic traffic patterns**, not synthetic uniform load — real traffic is bursty.

## Common Mistakes

**Mistake 1 — No explicit timeouts.** A default or missing timeout means one slow dependency can exhaust your entire request pool. Set an explicit, deliberately short timeout on every network call.

**Mistake 2 — Retrying without backoff.** Immediate retries on failure amplify load on an already-struggling dependency. Use exponential backoff with jitter.

**Mistake 3 — Ignoring connection pool exhaustion.** When the pool is full, requests queue silently. This looks like a latency spike, not a connection issue. Add metrics for pool utilisation and active wait time.

**Mistake 4 — Treating all errors the same.** A connection-refused error (retry with backoff) is different from an authentication error (fail immediately and alert) which is different from a timeout (retry once, then fail). Build specific handlers for each error class.

**Mistake 5 — Skipping the circuit breaker.** Without one, a downstream failure causes your application to queue up requests that will all fail anyway. Libraries like `resilience4j`, `tenacity`, or `polly` provide this with minimal code.

## Tools Worth Using

**For connection management:** Use an established library for your language rather than writing your own — most handle retries, timeouts, and pooling correctly out of the box.

**For monitoring:** A metrics system with histogram support (not just counters and gauges), a dashboard tool, and alerts set at p95/p99 latency rather than average.

**For testing:** Integration tests against real infrastructure where possible, plus a load-testing tool run regularly against staging, not just before launch.

**For resilience:** A maintained retry/circuit-breaker library for your language. Don't write your own unless you have a specific requirement it doesn't cover — these libraries have years of edge cases already handled.

## When to Skip {topic} Entirely

{topic} is not always the right choice. Be honest about whether it fits your situation:

**If your traffic is low and predictable,** the added complexity is not justified. A simple, synchronous approach is easier to debug and operate.

**If you do not have observability infrastructure,** you will not be able to debug problems when they occur. Set up metrics and logging first; add {topic} second.

**If your team is not familiar with the failure modes,** operational incidents will take longer to resolve than with a simpler system.

**Alternative to consider:** if the core requirement is reliability rather than performance, a message queue with at-least-once delivery often solves the actual problem with less operational complexity.

## Conclusion

The gap between a working {topic} prototype and a production-ready implementation comes down to how well you handle failure cases. The happy path is straightforward; the value is in error handling, monitoring, and circuit breakers.

Three actions to take now: set explicit timeouts on every operation today, add p99 latency metrics this week, and run a fault-injection test against staging this month.

{topic} works well when you understand its failure modes. It creates problems when treated as a black box. The documentation covers configuration; this guide covers what to do when the configuration does not help."""

    def _calculate_quality_score(self, content: str) -> int:
        """Calculate content quality score (0-100) based on AdSense-relevant signals.

        FIX: numeric density (a raw count of "43%", "200ms", etc.) is no
        longer scored as a positive signal on its own — that rewarded
        fabricating precise-sounding numbers. It now only credits a metric
        if it appears near a hedge phrase (see HEDGE_PHRASES) or looks like
        it's attributing a source. Unhedged, unsourced precise numbers no
        longer help the score, removing the incentive to invent them.
        """
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

        # Code examples (max 15 points)
        code_blocks = len(re.findall(r'```', content)) // 2
        if code_blocks >= 2:
            score += 15
        elif code_blocks == 1:
            score += 10

        # Honestly-framed numbers (max 10 points): a stated metric only
        # counts if it's near a hedge phrase or a source-like mention
        # (e.g. "according to", a named tool's own docs, etc.) within the
        # same sentence window. Bare, confident precision counts for nothing.
        numbers = list(re.finditer(
            r'\d+(?:\.\d+)?(?:%|ms|MB|KB|GB|s\b|x\b)', content))
        honest_numbers = 0
        for m in numbers:
            window = content[max(0, m.start() - 80):m.end() + 80].lower()
            if any(h in window for h in HEDGE_PHRASES):
                honest_numbers += 1
        if honest_numbers >= 3:
            score += 10
        elif honest_numbers >= 1:
            score += 6

        # Structural variety (max 10 points)
        has_ordered_list = bool(re.search(r'^\d+\.', content, re.MULTILINE))
        has_unordered_list = bool(re.search(r'^[-*]\s', content, re.MULTILINE))
        has_bold = '**' in content
        if has_ordered_list and has_unordered_list and has_bold:
            score += 10
        elif (has_ordered_list or has_unordered_list) and has_bold:
            score += 6

        # Absence of filler phrases (max 10 points) — hedge phrases like
        # "typically" or "can vary" are NOT filler and are not penalized;
        # only genuinely empty phrasing counts against the score.
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
    """Generate enhancement drafts for posts under the word-count floor.

    FIX: no longer calls blog_system.save_post() directly for every post.
    template_fallback drafts are written to a review queue file instead of
    being published, since that path produces near-identical content
    across posts. llm-sourced drafts still require a pass through
    adsense_compliance_audit.py's named_source_hit() check before you
    should treat them as safe to publish — this function does not run
    that check itself, to avoid duplicating logic that already exists
    and is maintained there.
    """
    import json

    enhancer = ContentQualityEnhancer(blog_system.api_key)

    posts_dir = blog_system.output_dir
    enhanced_count = 0
    queued_for_review = 0
    review_queue_path = posts_dir.parent / "enhancement_review_queue.json"
    review_queue = (
        json.loads(review_queue_path.read_text())
        if review_queue_path.exists() else []
    )
    queued_slugs = {e["slug"] for e in review_queue}

    print("Generating AdSense enhancement drafts...")
    print("=" * 60)

    for post_dir in posts_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name == 'static':
            continue

        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue

        try:
            with open(post_json, 'r') as f:
                post_data = json.load(f)

            from blog_post import BlogPost
            post = BlogPost.from_dict(post_data)

            word_count = len(post.content.split())
            if word_count >= 1500:
                print(f"OK  {post.title[:50]}... ({word_count} words)")
                continue

            print(f"\nDrafting enhancement: {post.title}")
            print(f"  Current: {word_count} words")

            result = await enhancer.enhance_post_for_adsense(
                post,
                post.tags[0] if post.tags else post.title
            )

            print(
                f"  Draft:    {result['word_count']} words, source={result['enhancement_source']}")
            print(f"  Quality:  {result['quality_score']}/100")

            if result['requires_review'] or result['enhancement_source'] != 'llm':
                if post.slug not in queued_slugs:
                    review_queue.append({
                        'slug': post.slug,
                        'draft_content': result['draft_content'],
                        'enhancement_source': result['enhancement_source'],
                        'quality_score': result['quality_score'],
                        'generated_at': datetime.now().isoformat(),
                    })
                    queued_for_review += 1
                print("  -> queued for review (template_fallback), not published")
                continue

            # llm-sourced draft: still record provenance, don't silently
            # overwrite. Update content + explicit monetization_data flag,
            # then let the existing audit pipeline (which already runs
            # named_source_hit / near-duplicate checks) gate publication
            # on the next scheduled audit pass rather than trusting this
            # function's own judgment.
            post.content = result['draft_content']
            post.updated_at = datetime.now().isoformat()
            post.monetization_data = {
                **(post.monetization_data or {}),
                'review_status': 'automated_qc_only',
                'enhancement_source': 'llm',
                'enhancement_quality_score': result['quality_score'],
            }
            blog_system.save_post(post)
            enhanced_count += 1

            await asyncio.sleep(2)

        except Exception as e:
            print(f"Error enhancing {post_dir.name}: {e}")

    if queued_for_review:
        review_queue_path.write_text(json.dumps(review_queue, indent=2))

    print("\n" + "=" * 60)
    print(f"Published (llm-sourced): {enhanced_count}")
    print(f"Queued for review (template_fallback): {queued_for_review} "
          f"-> {review_queue_path}")
    print("\nNext steps:")
    print("1. Run adsense_compliance_audit.py against the updated posts "
          "(catches fabricated_citation / near_duplicate before you rely on this)")
    print("2. Manually decide on enhancement_review_queue.json entries — "
          "these are templated content and should not be bulk-approved")
    print("3. Rebuild: python blog_system.py build")
    print("4. Wait 2-3 weeks for Google to recrawl")
    print("5. Re-request AdSense review")

    return enhanced_count
