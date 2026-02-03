#!/usr/bin/env python3
"""
NUCLEAR FIX for Blog Post Header Styling
This script patches static_site_generator.py to ensure correct HTML structure
"""

import re

# Read the current static_site_generator.py
with open('static_site_generator.py', 'r') as f:
    content = f.read()

# The CORRECT post template with proper header structure
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
# Pattern to match the post template definition
pattern = r'"post":\s*"""<!DOCTYPE html>.*?</html>""",'

# Replace with the new template
new_content = re.sub(pattern, NEW_POST_TEMPLATE, content, flags=re.DOTALL)

# Verify the replacement worked
if new_content != content:
    # Write back
    with open('static_site_generator.py', 'w') as f:
        f.write(new_content)
    print("✅ SUCCESS: static_site_generator.py has been patched!")
    print("\nNOW RUN:")
    print("  python blog_system.py build")
    print("  git add .")
    print("  git commit -m 'Fix: Ensure post header has class=post-header'")
    print("  git push")
else:
    print("❌ ERROR: Could not find post template to replace")
    print("Manual fix required - see MANUAL_FIX.md")