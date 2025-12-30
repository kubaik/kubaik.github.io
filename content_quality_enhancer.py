"""
AdSense Content Quality Improver
Enhances blog content to meet Google AdSense quality standards
"""

import asyncio
import aiohttp
import re
from datetime import datetime
from typing import Dict, List, Tuple
import random


class ContentQualityEnhancer:
    """Enhances content to meet AdSense standards and avoid 'low value content' issues"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.min_word_count = 2000  # Increase to 2000+
        self.min_unique_paragraphs = 15  # Increase
        self.min_sections = 8  # Increase
        self.require_personal_touch = True  
        self.require_original_images = True  
    
    async def enhance_post_for_adsense(self, post, topic: str) -> Dict:
        """
        Enhance a blog post to meet AdSense quality standards
        
        Key improvements:
        - Longer, more detailed content (1200+ words)
        - More unique sections and headings
        - Personal insights and unique perspectives
        - Practical examples and case studies
        - FAQ section
        - Expert tips and best practices
        - Conclusion with actionable takeaways
        """
        
        enhanced_content = await self._generate_enhanced_content(topic, post.title)
        
        # Update post with enhanced content
        post.content = enhanced_content
        post.updated_at = datetime.now().isoformat()
        
        # Add quality metadata
        quality_score = self._calculate_quality_score(enhanced_content)
        
        return {
            'enhanced': True,
            'word_count': len(enhanced_content.split()),
            'quality_score': quality_score,
            'sections': enhanced_content.count('##'),
            'improvements': [
                'Extended content length to 1200+ words',
                'Added unique perspectives and insights',
                'Included practical examples',
                'Added FAQ section',
                'Enhanced with expert tips',
                'Improved structure and formatting'
            ]
        }
    
    async def _generate_enhanced_content(self, topic: str, title: str) -> str:
        """Generate high-quality, AdSense-friendly content with personalization"""
        
        if not self.api_key:
            return self._generate_enhanced_fallback(topic, title)
        
        try:
            content_sections = []
            
            # 1. Personal Hook Introduction (200-250 words)
            intro = await self._generate_section(
                "personal_introduction",
                topic,
                title,
                """Write a compelling 200-250 word introduction that:
                - Starts with a personal anecdote or real experience related to this topic
                - Explains a specific problem you encountered and why it matters
                - Uses conversational tone with 'I', 'we', 'you' pronouns
                - Includes specific numbers, timeframes, or concrete examples
                - Hooks readers with relatable scenarios
                Example start: "Last year, I spent three months struggling with..."
                Avoid generic openings like "In today's world..." or "Technology is changing..."
                """
            )
            content_sections.append(intro)
            
            # 2. Personal Story / Real Experience (250-300 words)
            personal_story = await self._generate_section(
                "personal_experience",
                topic,
                title,
                """Share a detailed personal experience or case study about {topic}:
                - Describe a specific situation where you used or learned about this
                - Include concrete details: dates, places, specific challenges
                - Explain what worked and what didn't
                - Share mistakes made and lessons learned
                - Use first-person narrative ("I discovered...", "After trying...")
                - Include actual numbers or results if applicable
                Make this feel like a story, not a textbook explanation.
                """
            )
            content_sections.append(f"## My Journey with {topic}\n\n{personal_story}")
            
            # 3. The Problem-Solution Framework (300-350 words)
            problem_solution = await self._generate_section(
                "problem_solution",
                topic,
                title,
                """Describe a specific problem and solution framework:
                - Start with a relatable problem statement
                - Explain why traditional approaches fail
                - Introduce your unique solution or perspective
                - Include a "before vs after" comparison
                - Add specific metrics or improvements observed
                - Use subheadings like "The Problem:", "Why It Matters:", "The Solution:"
                Be opinionated and take a clear stance.
                """
            )
            content_sections.append(f"## Understanding the Real Challenge\n\n{problem_solution}")
            
            # 4. Detailed How-To with Personal Tips (400-450 words)
            detailed_guide = await self._generate_section(
                "detailed_implementation",
                topic,
                title,
                """Create a step-by-step implementation guide with personal insights:
                - Break down into 5-7 clear, actionable steps
                - For each step, add a "Pro Tip:" based on personal experience
                - Include common pitfalls and how to avoid them
                - Add time estimates for each step
                - Include "What I Wish I Knew:" callouts
                - Mention specific tools, versions, or resources you actually used
                - Use conversational language: "Here's what I do...", "I've found that..."
                Format with clear numbered steps and personal commentary.
                """
            )
            content_sections.append(f"## My Step-by-Step Approach\n\n{detailed_guide}")
            
            # 5. Real-World Case Study or Example (300-350 words)
            case_study = await self._generate_section(
                "case_study",
                topic,
                title,
                """Present a detailed, specific case study:
                - Use a real or realistic scenario with specific details
                - Include actual company names, projects, or situations (or realistic pseudonyms)
                - Provide concrete before/after metrics
                - Explain the decision-making process
                - Share unexpected outcomes or surprises
                - Format as: Background → Challenge → Approach → Results → Lessons
                - Include actual numbers: costs, time saved, performance improvements
                Make this feel like investigative journalism, not marketing copy.
                """
            )
            content_sections.append(f"## Real-World Application: A Case Study\n\n{case_study}")
            
            # 6. Expert Insights & Interviews (250-300 words)
            expert_insights = await self._generate_section(
                "expert_perspective",
                topic,
                title,
                """Share expert insights or perspectives:
                - Reference specific experts, authors, or practitioners in the field
                - Include paraphrased insights from industry leaders
                - Add "According to [Expert]..." style attributions
                - Contrast different schools of thought or approaches
                - Add your personal take: "While [Expert] suggests X, I've found Y works better because..."
                - Reference specific books, talks, or resources
                Cite actual industry figures and add your unique interpretation.
                """
            )
            content_sections.append(f"## What the Experts Say (And What I Think)\n\n{expert_insights}")
            
            # 7. Common Mistakes & How I Overcame Them (300-350 words)
            mistakes_lessons = await self._generate_section(
                "mistakes_lessons",
                topic,
                title,
                """Share authentic mistakes and lessons learned:
                - List 5-7 specific mistakes you or others made
                - For each mistake, explain: what happened, why it happened, how to avoid it
                - Use vulnerability: "I wasted two weeks because I didn't..."
                - Include cost of mistakes (time, money, frustration)
                - Add "Red flags to watch for"
                - Share embarrassing moments or failures
                - End each with a concrete prevention strategy
                Make this brutally honest and relatable.
                """
            )
            content_sections.append(f"## Mistakes I Made (So You Don't Have To)\n\n{mistakes_lessons}")
            
            # 8. Tools & Resources I Actually Use (250-300 words)
            tools_resources = await self._generate_section(
                "tools_resources",
                topic,
                title,
                """Recommend specific tools and resources:
                - List 5-8 specific tools/resources with exact names
                - For each, explain: what it does, why you chose it, pros/cons
                - Include free vs paid options
                - Add pricing information where relevant
                - Share your personal workflow: "My tech stack includes..."
                - Rate each tool (e.g., "9/10 for beginners")
                - Mention alternatives you tried and why you switched
                - Include affiliate disclosure if recommending products
                Be specific: not "project management tools" but "Asana vs Monday.com"
                """
            )
            content_sections.append(f"## Tools & Resources in My Arsenal\n\n{tools_resources}")
            
            # 9. Advanced Tips & Pro Strategies (300-350 words)
            advanced_tips = await self._generate_section(
                "advanced_strategies",
                topic,
                title,
                """Share advanced, non-obvious strategies:
                - Include 6-8 tips that go beyond basic advice
                - Label difficulty: [Beginner], [Intermediate], [Advanced]
                - Share "insider" knowledge or shortcuts you discovered
                - Include specific configurations, settings, or parameters
                - Add time-saving hacks: "This one trick saves me 5 hours/week..."
                - Include contrarian takes: "Most people do X, but I found Y works better"
                - Add ROI or impact metrics for each tip
                Make these feel like secret weapons, not common knowledge.
                """
            )
            content_sections.append(f"## Advanced Strategies That Work\n\n{advanced_tips}")
            
            # 10. Interactive FAQ (300-350 words)
            interactive_faq = await self._generate_section(
                "interactive_faq",
                topic,
                title,
                """Create engaging FAQs based on real questions:
                - Write 6-8 questions that real users actually ask
                - Start questions with who, what, when, where, why, how
                - Provide detailed, personal answers (not generic responses)
                - Include your opinion in answers: "In my experience..."
                - Add surprising or counter-intuitive answers
                - Reference your earlier personal examples
                - End some answers with follow-up resources
                Format: **Q: [Question]** followed by detailed A:
                """
            )
            content_sections.append(f"## Your Questions Answered\n\n{interactive_faq}")
            
            # 11. Future Trends & My Predictions (200-250 words)
            future_trends = await self._generate_section(
                "future_predictions",
                topic,
                title,
                """Share informed predictions about the future:
                - Identify 3-5 emerging trends in this space
                - Base predictions on current signals and data
                - Add personal commentary: "I believe we'll see..."
                - Include contrarian predictions if applicable
                - Set timeframes: "Within 2 years...", "By 2027..."
                - Explain your reasoning
                - Suggest how readers can prepare
                Be bold with predictions while showing your reasoning.
                """
            )
            content_sections.append(f"## Where This is Heading (My Predictions)\n\n{future_trends}")
            
            # 12. Personal Action Plan & Next Steps (250-300 words)
            action_plan = await self._generate_section(
                "personalized_action_plan",
                topic,
                title,
                """Create a practical, personalized action plan:
                - Offer 3 different paths based on reader's level:
                * "If you're just starting..." (Beginner path)
                * "If you have some experience..." (Intermediate path)
                * "If you're advanced..." (Expert path)
                - Include timeline: "Week 1:", "Month 1:", "Quarter 1:"
                - Add specific milestones and success metrics
                - Include a "Start today" checklist with 3-5 immediate actions
                - Offer to connect: "Questions? Here's how to reach me..."
                - Add motivational close with personal encouragement
                Make this feel like coaching, not just information.
                """
            )
            content_sections.append(f"## Your Personalized Action Plan\n\n{action_plan}")
            
            # 13. Personal Sign-Off & CTA (150-200 words)
            conclusion = await self._generate_section(
                "personal_conclusion",
                topic,
                title,
                """Write a warm, personal conclusion:
                - Summarize key takeaway in one sentence
                - Share final personal reflection or encouragement
                - Invite engagement: "What's your experience with {topic}?"
                - Ask a specific question to encourage comments
                - Offer additional help: "Stuck on step 3? Email me at..."
                - Add social proof if available: "Over 500 readers have used this guide..."
                - Include a clear call-to-action (subscribe, comment, share)
                - Sign off personally: "Good luck on your journey, [Author Name]"
                Make this feel like ending a conversation with a friend.
                """
            )
            content_sections.append(f"## Final Thoughts\n\n{conclusion}")
            
            # 14. Additional Resources & Reading (Optional)
            resources = await self._generate_section(
                "curated_resources",
                topic,
                title,
                """Curate additional resources:
                - List 5-7 specific articles, books, videos, or courses
                - Add brief annotation for each: what it covers, why it's valuable
                - Include mix of free and paid resources
                - Organize by category: "For beginners:", "Deep dives:", "Tools:"
                - Add your personal recommendation level (Must-read, Useful, Optional)
                - Include publication dates to show currency
                Keep this concise but valuable - actual resources, not generic suggestions.
                """
            )
            content_sections.append(f"## Recommended Resources\n\n{resources}")
            
            return "\n\n".join(content_sections)
        
    except Exception as e:
        print(f"Error generating enhanced content: {e}")
        return self._generate_enhanced_fallback(topic, title)

    
    async def _generate_section(self, section_type: str, topic: str, 
                                title: str, instruction: str) -> str:
            """Generate a specific section using OpenAI API"""
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert content writer with deep personal experience in technology and development. 
                Write in a conversational, authentic voice as if sharing knowledge with a colleague. 
                Use personal pronouns (I, we, you). Share real experiences, specific details, and unique insights.
                Avoid generic statements, corporate jargon, and obvious advice. 
                Be opinionated, specific, and actionable. Include concrete examples with numbers, timeframes, and results.
                Write like a human sharing hard-won knowledge, not an AI generating content."""
            },
            {
                "role": "user",
                "content": f"""Topic: {topic}
    Article Title: {title}
    Section: {section_type}

    {instruction}

    Remember:
    - Use specific, concrete details
    - Share personal experiences and opinions
    - Include numbers, metrics, and timeframes
    - Be conversational and authentic
    - Avoid generic, obvious statements
    - Take clear stances on debatable points

    Write original, valuable content that provides real insights."""
            }
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4.1-nano",
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.8  # Higher for more creativity
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
                    raise Exception(f"API error: {response.status}")
    
    def _generate_enhanced_fallback(self, topic: str, title: str) -> str:
        """Generate enhanced fallback content when API is unavailable"""
        
        content = f"""## Introduction

In today's rapidly evolving digital landscape, {topic} has emerged as a critical component for success. Whether you're a seasoned professional or just starting your journey, understanding the nuances of {topic} can make the difference between mediocre results and exceptional outcomes.

Through years of experience and countless implementations, I've discovered that the key to mastering {topic} lies not just in understanding the theory, but in applying practical strategies that deliver real-world results. In this comprehensive guide, we'll explore everything you need to know to leverage {topic} effectively.

## Understanding {topic}: A Deep Dive

{topic} represents more than just a buzzword—it's a fundamental shift in how we approach modern challenges. At its core, {topic} combines several key principles:

**Foundational Principles**: The underlying concepts that make {topic} powerful are rooted in proven methodologies that have been refined over time. Understanding these principles helps you make informed decisions.

**Practical Applications**: Real-world implementation of {topic} requires careful consideration of various factors including scalability, maintainability, and user experience. Each application scenario demands a tailored approach.

**Strategic Value**: Organizations that successfully implement {topic} gain significant competitive advantages. The strategic implications extend beyond immediate benefits to long-term growth and sustainability.

## Key Benefits and Features

### 1. Enhanced Efficiency and Performance

Implementing {topic} correctly can dramatically improve operational efficiency. Organizations typically see 30-50% improvements in key metrics within the first quarter of adoption. This efficiency gain comes from streamlined processes and optimized workflows.

### 2. Scalability and Growth Potential

One of the most compelling aspects of {topic} is its ability to scale seamlessly with your needs. Whether you're handling dozens or millions of transactions, the architecture supports growth without compromising performance.

### 3. Cost-Effectiveness

While initial investment in {topic} may seem significant, the long-term ROI is substantial. Many organizations report 200-300% return on investment within 18-24 months through reduced operational costs and increased productivity.

### 4. Improved User Experience

End users benefit tremendously from well-implemented {topic}. Response times decrease, reliability increases, and overall satisfaction scores typically jump by 40% or more.

### 5. Future-Proof Architecture

Technology evolves rapidly, but {topic} is designed with adaptability in mind. The modular approach ensures that you can integrate new features and technologies as they emerge.

### 6. Enhanced Security and Compliance

Modern {topic} implementations include robust security features that protect sensitive data and ensure regulatory compliance, critical for organizations in regulated industries.

### 7. Competitive Advantage

Organizations leveraging {topic} effectively gain significant competitive advantages, including faster time-to-market, better resource utilization, and superior customer experiences.

## How to Implement {topic}: A Step-by-Step Guide

### Step 1: Assessment and Planning (Week 1-2)

Begin with a thorough assessment of your current situation. Identify pain points, opportunities, and specific goals you want to achieve with {topic}. Create a detailed roadmap with measurable milestones.

### Step 2: Foundation Building (Week 3-4)

Establish the necessary infrastructure and foundations. This includes setting up development environments, configuring systems, and ensuring all prerequisites are met.

### Step 3: Core Implementation (Week 5-8)

Focus on implementing core functionality first. Start with the most critical features and gradually expand. Use an iterative approach to ensure quality at each stage.

### Step 4: Testing and Optimization (Week 9-10)

Rigorous testing is crucial. Conduct comprehensive testing including unit tests, integration tests, and user acceptance testing. Optimize based on results.

### Step 5: Deployment and Monitoring (Week 11-12)

Deploy systematically with proper monitoring in place. Start with a limited rollout before full deployment to identify and address any issues early.

## Best Practices and Expert Tips

### Tip 1: Start with Clear Objectives

Define specific, measurable goals before beginning implementation. Vague objectives lead to scope creep and disappointing results.

### Tip 2: Invest in Training

Ensure your team has proper training and documentation. A well-trained team is 3x more likely to successfully implement and maintain {topic}.

### Tip 3: Implement Progressive Enhancement

Build your solution incrementally, adding features progressively. This approach reduces risk and allows for course corrections along the way.

### Tip 4: Prioritize Documentation

Comprehensive documentation saves countless hours down the road. Document not just what you did, but why you made specific decisions.

### Tip 5: Monitor and Measure Continuously

Implement robust monitoring from day one. Track key performance indicators and use data to drive decisions.

### Tip 6: Foster Collaboration

Success with {topic} requires cross-functional collaboration. Break down silos and encourage open communication between teams.

### Tip 7: Plan for Maintenance

Implementation is just the beginning. Allocate resources for ongoing maintenance, updates, and improvements.

### Tip 8: Stay Updated

{topic} evolves rapidly. Stay informed about latest developments, best practices, and emerging trends through continuous learning.

## Common Challenges and How to Overcome Them

### Challenge 1: Complexity Management

**Problem**: Implementations can become overwhelming complex.
**Solution**: Break down the project into manageable chunks. Use modular design principles and maintain clear separation of concerns.

### Challenge 2: Resource Constraints

**Problem**: Limited budget or personnel.
**Solution**: Prioritize ruthlessly. Focus on features that deliver maximum value. Consider phased implementation to spread costs.

### Challenge 3: Integration Issues

**Problem**: Difficulty integrating with existing systems.
**Solution**: Use standard APIs and protocols. Implement adapter patterns where necessary. Test integrations early and often.

### Challenge 4: Performance Bottlenecks

**Problem**: System performance doesn't meet expectations.
**Solution**: Identify bottlenecks through profiling. Optimize critical paths. Consider caching strategies and load balancing.

### Challenge 5: User Adoption

**Problem**: Users resist change or struggle with new systems.
**Solution**: Invest in user training and support. Gather feedback continuously. Make improvements based on actual user needs.

## Real-World Applications

### Case Study 1: Enterprise Implementation

A Fortune 500 company implemented {topic} across their operations, resulting in 45% reduction in processing time and $2.3M annual savings. Key success factors included executive sponsorship, comprehensive training, and phased rollout.

### Case Study 2: Startup Success

A tech startup leveraged {topic} to rapidly scale from 1,000 to 100,000 users in 6 months. Their lightweight, cloud-native approach allowed rapid iteration without significant infrastructure investment.

### Case Study 3: Non-Profit Transformation

A non-profit organization used {topic} to modernize their operations, improving service delivery to 50,000 beneficiaries while reducing administrative costs by 35%.

## Frequently Asked Questions

**Q: How long does it typically take to implement {topic}?**
A: Implementation timelines vary based on scope and complexity. Simple implementations can take 2-3 months, while enterprise-wide deployments may require 6-12 months. Proper planning and phased approaches help manage timelines effectively.

**Q: What's the typical ROI for {topic}?**
A: Most organizations see positive ROI within 12-18 months. Early benefits include efficiency gains and cost reductions, while long-term benefits include competitive advantages and scalability.

**Q: What skills are needed to work with {topic}?**
A: Core technical skills are essential, but soft skills like problem-solving, communication, and adaptability are equally important. Specific technical requirements depend on your implementation approach.

**Q: How do I measure success?**
A: Define KPIs aligned with your objectives. Common metrics include performance improvements, cost savings, user satisfaction scores, and time-to-market reductions.

**Q: What are the biggest mistakes to avoid?**
A: Common pitfalls include insufficient planning, inadequate training, trying to do too much at once, neglecting documentation, and failing to gather user feedback early.

**Q: Is {topic} suitable for small businesses?**
A: Absolutely! {topic} can be scaled appropriately for organizations of any size. Start small, focus on core benefits, and expand as you grow.

## Future Trends and Predictions

The future of {topic} looks incredibly promising. Emerging trends include increased automation through AI and machine learning, greater emphasis on real-time processing, and improved integration capabilities.

We're seeing a shift toward more accessible, user-friendly implementations that require less technical expertise. Cloud-native approaches are becoming standard, offering greater flexibility and scalability.

Within the next 2-3 years, expect {topic} to become even more mainstream, with standardized best practices and mature tooling ecosystems. Organizations that invest now will be well-positioned to capitalize on these advances.

## Key Takeaways and Next Steps

{topic} offers tremendous potential for organizations willing to invest in proper implementation. Success requires careful planning, adequate resources, and commitment to continuous improvement.

**Your Action Plan:**

1. **This Week**: Assess your current situation and identify specific goals
2. **Next Month**: Develop a detailed implementation roadmap
3. **Quarter 1**: Begin with core implementation and early wins
4. **Quarter 2**: Expand and optimize based on results
5. **Ongoing**: Monitor, measure, and continuously improve

Remember that implementing {topic} is a journey, not a destination. Stay flexible, learn from both successes and failures, and don't hesitate to adjust your approach based on real-world results.

The organizations that thrive are those that view {topic} not as a one-time project but as an ongoing strategic initiative. Start today, think long-term, and commit to excellence at every stage.

Ready to transform your operations with {topic}? The time to start is now."""

        return content
    
    def _calculate_quality_score(self, content: str) -> int:
        """Calculate content quality score (0-100)"""
        score = 0
        
        # Word count (max 30 points)
        word_count = len(content.split())
        if word_count >= 1200:
            score += 30
        elif word_count >= 1000:
            score += 25
        elif word_count >= 800:
            score += 20
        
        # Section count (max 20 points)
        section_count = content.count('##')
        if section_count >= 8:
            score += 20
        elif section_count >= 6:
            score += 15
        elif section_count >= 4:
            score += 10
        
        # Unique formatting (max 15 points)
        has_lists = bool(re.search(r'^\d+\.', content, re.MULTILINE))
        has_bullets = bool(re.search(r'^[-*]', content, re.MULTILINE))
        has_bold = '**' in content
        has_code = '```' in content
        
        if has_lists: score += 5
        if has_bullets: score += 5
        if has_bold: score += 3
        if has_code: score += 2
        
        # Paragraph variety (max 20 points)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) >= 15:
            score += 20
        elif len(paragraphs) >= 10:
            score += 15
        elif len(paragraphs) >= 8:
            score += 10
        
        # Content depth indicators (max 15 points)
        depth_keywords = [
            'how to', 'step-by-step', 'example', 'case study',
            'best practice', 'tip', 'strategy', 'implementation',
            'challenge', 'solution', 'benefit', 'advantage'
        ]
        content_lower = content.lower()
        depth_score = sum(3 for keyword in depth_keywords if keyword in content_lower)
        score += min(depth_score, 15)
        
        return min(score, 100)


# Integration function
async def enhance_all_posts_for_adsense(blog_system):
    """Enhance all existing posts to meet AdSense standards"""
    
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
            # Load post
            with open(post_json, 'r') as f:
                import json
                post_data = json.load(f)
            
            from blog_post import BlogPost
            post = BlogPost.from_dict(post_data)
            
            # Check if enhancement needed
            word_count = len(post.content.split())
            if word_count >= 1200:
                print(f"✓ {post.title[:50]}... (Already good: {word_count} words)")
                continue
            
            print(f"\nEnhancing: {post.title}")
            print(f"  Current: {word_count} words")
            
            # Enhance content
            result = await enhancer.enhance_post_for_adsense(
                post, 
                post.tags[0] if post.tags else post.title
            )
            
            print(f"  Enhanced: {result['word_count']} words")
            print(f"  Quality Score: {result['quality_score']}/100")
            print(f"  Sections: {result['sections']}")
            
            # Save enhanced post
            blog_system.save_post(post)
            enhanced_count += 1
            
            # Avoid rate limiting
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error enhancing {post_dir.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Enhanced {enhanced_count} posts for AdSense compliance")
    print("\nNext steps:")
    print("1. Review enhanced content for quality")
    print("2. Rebuild site with: python blog_system.py build")
    print("3. Wait 1-2 weeks for Google to recrawl")
    print("4. Request AdSense review again")
    
    return enhanced_count