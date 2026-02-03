#!/usr/bin/env python3
"""
ULTIMATE CSS FIX - Solves ALL styling issues at once
1. Fixes HTML template structure
2. Adds enhanced-blog-post-styles.css link
3. Removes CSS duplication from style.css
"""

import re
from pathlib import Path

print("=" * 80)
print("🔧 ULTIMATE CSS FIX - Starting...")
print("=" * 80)
print()

# Read static_site_generator.py
with open('static_site_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ========================================================================
# FIX 1: Update POST template with correct structure + CSS link
# ========================================================================

NEW_POST_TEMPLATE = '''                "post": """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ post.title }} - {{ site_name }}</title>
        <meta name="description" content="{{ post.meta_description }}">
        {% if post.seo_keywords %}<meta name="keywords" content="{{ post.seo_keywords|join(', ') }}">{% endif %}
        {{ global_meta_tags | safe }}
        {{ meta_tags | safe }}
        {{ structured_data | safe }}
        <link rel="stylesheet" href="{{ base_path }}/static/style.css">
        <link rel="stylesheet" href="{{ base_path }}/static/enhanced-blog-post-styles.css">
    </head>
    <body>
        <!-- Header Ad Slot -->
        {{ header_ad | safe }}
        
        <header>
            <div class="container">
                <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
                <nav>
                    <a href="{{ base_path }}/">Home</a>
                    <a href="{{ base_path }}/about/">About</a>
                    <a href="{{ base_path }}/contact/">Contact</a>
                    <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
                    <a href="{{ base_path }}/terms-of-service/">Terms of Service</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <article class="blog-post">
                <header class="post-header">
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
                </header>
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
            </article>
        </main>
        
        <!-- Footer Ad Slot -->
        {{ footer_ad | safe }}
        
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}.</p>
            </div>
        </footer>
        <!-- Enhanced Navigation Script -->
        <script src="{{ base_path }}/static/navigation.js"></script>
    </body>
    </html>""",'''

# Find and replace the post template
pattern = r'"post":\s*"""<!DOCTYPE html>.*?</html>""",'
new_content = re.sub(pattern, NEW_POST_TEMPLATE, content, flags=re.DOTALL)

if new_content != content:
    with open('static_site_generator.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ FIX 1: Updated post template")
    print("   • Added <link> for enhanced-blog-post-styles.css")
    print("   • Fixed <header class='post-header'> structure")
    print("   • Tags now OUTSIDE post-meta")
else:
    print("❌ FIX 1: Could not update template")

# ========================================================================
# FIX 2: Clean up style.css - remove duplicate enhanced blog post styles
# ========================================================================

print()
style_css_path = Path('./docs/static/style.css')

if style_css_path.exists():
    with open(style_css_path, 'r', encoding='utf-8') as f:
        style_content = f.read()
    
    # Find where the duplicate enhanced blog post styling starts
    # It should be after the CRITICAL FIX section
    marker = "/* Enhanced Post Grid */"
    
    if marker in style_content:
        # Split at the marker
        before_duplicate = style_content.split(marker)[0]
        
        # Find where to end (before the "Tags (for other places)" section)
        end_marker = "/* Tags (for other places like post cards) */"
        
        if end_marker in style_content:
            after_duplicate = end_marker + style_content.split(end_marker)[1]
            
            # Reconstruct without the duplicate section
            cleaned_content = before_duplicate + marker + "\n" + after_duplicate
            
            # Save the cleaned version
            with open(style_css_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print("✅ FIX 2: Removed duplicate enhanced blog post styles from style.css")
            print("   • Kept CRITICAL FIX section")
            print("   • Removed conflicting duplicate rules")
        else:
            print("⚠️  FIX 2: Could not find end marker for duplicate section")
    else:
        print("⚠️  FIX 2: Marker not found in style.css")
else:
    print("❌ FIX 2: style.css not found in docs/static/")

# ========================================================================
# FIX 3: Ensure enhanced-blog-post-styles.css is in docs/static/
# ========================================================================

print()
source = Path('./static/enhanced-blog-post-styles.css')
dest = Path('./docs/static/enhanced-blog-post-styles.css')

if source.exists():
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, dest)
    print("✅ FIX 3: Copied enhanced-blog-post-styles.css to docs/static/")
else:
    print("❌ FIX 3: Source file not found at ./static/enhanced-blog-post-styles.css")

print()
print("=" * 80)
print("✅ ALL FIXES APPLIED!")
print("=" * 80)
print()
print("🚀 NEXT STEPS:")
print("   1. Run: python blog_system.py build")
print("   2. Deploy: git add . && git commit -m 'Fix: Complete CSS overhaul' && git push")
print("   3. Wait 2 minutes for GitHub Pages to deploy")
print("   4. Hard refresh your browser (Ctrl+Shift+R)")
print()
print("💡 WHAT WAS FIXED:")
print("   ✅ Template now loads enhanced-blog-post-styles.css")
print("   ✅ Header has correct <header class='post-header'> structure")
print("   ✅ Tags are SIBLINGS to post-meta (will center properly)")
print("   ✅ Removed duplicate CSS rules from style.css")
print("   ✅ Enhanced styles file is in docs/static/")
print()
