#!/usr/bin/env python3
"""
Quick fix for the post template structure
Ensures tags are OUTSIDE post-meta div
"""

import re

print("🔧 Fixing post template structure...")
print()

# Read static_site_generator.py
with open('static_site_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The CORRECT post template structure
CORRECT_HEADER = '''                <header class="post-header">
                    <h1>{{ post.title }}</h1>
                    <div class="post-meta">
                        <time datetime="{{ post.created_at }}">{{ post.created_at.split('T')[0] }}</time>
                    </div>
                    {% if post.tags %}
                    <div class="tags">
                        {% for tag in post.tags[:6] %}
                        <span class="tag">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                </header>'''

# Check current structure
if '<header class="post-header">' in content:
    print("✅ Template has <header class='post-header'>")
else:
    print("❌ Template missing <header class='post-header'>")

# Check if tags are likely inside post-meta (wrong)
# This is a heuristic check
if 'post-meta' in content and 'tags' in content:
    # Extract the post template
    template_match = re.search(r'<article class="blog-post">(.*?)</article>', content, re.DOTALL)
    if template_match:
        template_section = template_match.group(1)
        
        # Check for the problematic pattern
        if re.search(r'<div class="post-meta">.*?<div class="tags">', template_section, re.DOTALL):
            print("❌ Tags appear to be INSIDE post-meta div (this causes left-alignment!)")
            print()
            print("Fixing...")
            
            # Find and replace the article section
            pattern = r'(<article class="blog-post">)(.*?)(</article>)'
            
            def replace_article(match):
                before = match.group(1)
                after = match.group(3)
                
                # Replace with correct structure
                new_middle = '\n' + CORRECT_HEADER + '''
                <div class="post-content">
                    {{ post.content_html | safe }}
                    
                    <!-- Middle Ad Slot -->
                    {{ middle_ad | safe }}
                </div>
                
                <!-- Affiliate Disclaimer -->
                {% if post.affiliate_links %}
                <div class="affiliate-disclaimer">
                    <p><em>This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no additional cost to you.</em></p>
                </div>
                {% endif %}
            '''
                
                return before + new_middle + after
            
            new_content = re.sub(pattern, replace_article, content, flags=re.DOTALL)
            
            # Save
            with open('static_site_generator.py', 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ Fixed! Tags are now OUTSIDE post-meta")
            print()
            print("Structure is now:")
            print("  <header class='post-header'>")
            print("    <h1>Title</h1>")
            print("    <div class='post-meta'><time>Date</time></div>")
            print("    <div class='tags'>Tags...</div>  ← OUTSIDE post-meta!")
            print("  </header>")
            
        else:
            print("✅ Tags appear to be OUTSIDE post-meta div (correct!)")
else:
    print("⚠️  Could not verify tag position")

print()
print("=" * 80)
print("🚀 NEXT STEPS:")
print("=" * 80)
print("1. Run: python blog_system.py build")
print("2. Check a generated HTML file in docs/[post-name]/index.html")
print("3. Look for this structure:")
print("   <header class='post-header'>")
print("     <h1>...</h1>")
print("     <div class='post-meta'><time>...</time></div>")
print("     <div class='tags'>...</div>  ← Should be OUTSIDE post-meta")
print("   </header>")
print()
print("4. If correct, deploy:")
print("   git add . && git commit -m 'Fix: Correct post header structure' && git push")
print()
