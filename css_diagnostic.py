#!/usr/bin/env python3
"""
COMPREHENSIVE CSS DIAGNOSTIC
Analyzes why enhanced blog post styles aren't working
"""

import re
from pathlib import Path

print("=" * 80)
print("CSS DIAGNOSTIC TOOL")
print("=" * 80)
print()

# 1. Check if enhanced-blog-post-styles.css exists
enhanced_css = Path("./docs/static/enhanced-blog-post-styles.css")
if enhanced_css.exists():
    print("✅ enhanced-blog-post-styles.css EXISTS in docs/static/")
else:
    print("❌ enhanced-blog-post-styles.css NOT FOUND in docs/static/")
    print("   Location should be: ./docs/static/enhanced-blog-post-styles.css")

# 2. Check static/style.css for duplicate/conflicting rules
style_css = Path("./static/style.css")
if style_css.exists():
    with open(style_css, 'r', encoding='utf-8') as f:
        style_content = f.read()
    
    print()
    print("=" * 80)
    print("ANALYZING static/style.css")
    print("=" * 80)
    
    # Count occurrences of key selectors
    post_header_count = len(re.findall(r'\.post-header\s*{', style_content))
    print(f"\n📊 .post-header {{ ... }} appears: {post_header_count} times")
    
    if post_header_count > 3:
        print("   ⚠️  WARNING: Too many .post-header definitions!")
        print("   This causes CSS specificity conflicts")
    
    # Check for the CRITICAL FIX section
    if "CRITICAL FIX: POST HEADER CENTERING" in style_content:
        print("\n✅ Found 'CRITICAL FIX' section at top of style.css")
    else:
        print("\n❌ Missing 'CRITICAL FIX' section at top of style.css")
    
    # Check for duplicate enhanced blog post styling
    enhanced_sections = style_content.count("ENHANCED BLOG POST STYLING")
    print(f"\n📊 'ENHANCED BLOG POST STYLING' appears: {enhanced_sections} times")
    
    if enhanced_sections > 1:
        print("   ⚠️  WARNING: Enhanced blog post styles are duplicated!")
        print("   This causes conflicting CSS rules")
    
    # Check for syntax errors
    open_braces = style_content.count('{')
    close_braces = style_content.count('}')
    print(f"\n📊 CSS Syntax Check:")
    print(f"   Open braces {{ : {open_braces}")
    print(f"   Close braces }}: {close_braces}")
    
    if open_braces != close_braces:
        print(f"   ❌ SYNTAX ERROR: Mismatched braces (difference: {open_braces - close_braces})")
        print("   This breaks ALL CSS rules after the error!")
    else:
        print("   ✅ Braces are balanced")

# 3. Check if HTML is loading the CSS file
print()
print("=" * 80)
print("CHECKING HTML FOR CSS LINKS")
print("=" * 80)

# Find latest blog post
docs_dir = Path("./docs")
post_dirs = [d for d in docs_dir.iterdir() if d.is_dir() and d.name != 'static']

if post_dirs:
    post_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_post = post_dirs[0]
    
    html_file = latest_post / "index.html"
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        print(f"\n📄 Examining: {latest_post.name}/index.html")
        
        # Check for CSS links
        css_links = re.findall(r'<link[^>]*href="([^"]*\.css)"', html_content)
        print(f"\n📊 CSS files loaded:")
        for link in css_links:
            print(f"   • {link}")
        
        if not any('style.css' in link for link in css_links):
            print("\n   ❌ style.css is NOT linked!")
        else:
            print("\n   ✅ style.css is linked")
        
        if not any('enhanced-blog-post-styles.css' in link for link in css_links):
            print("   ❌ enhanced-blog-post-styles.css is NOT linked!")
            print("   → This is why enhanced styles don't work!")
        else:
            print("   ✅ enhanced-blog-post-styles.css is linked")

# 4. Check the actual HTML structure
print()
print("=" * 80)
print("HTML STRUCTURE ANALYSIS")
print("=" * 80)

if post_dirs and html_file.exists():
    # Find the article section
    article_match = re.search(r'<article class="blog-post">(.*?)</article>', html_content, re.DOTALL)
    if article_match:
        article_content = article_match.group(1)
        
        # Check header structure
        header_match = re.search(r'<(header|div)([^>]*)>(.*?)</(header|div)>', article_content, re.DOTALL)
        if header_match:
            tag_type = header_match.group(1)
            tag_attrs = header_match.group(2)
            
            print(f"\n📋 Post header structure:")
            print(f"   Tag type: <{tag_type}>")
            print(f"   Attributes: {tag_attrs.strip()}")
            
            if tag_type != 'header':
                print(f"   ❌ Using <{tag_type}> instead of <header>")
            else:
                print("   ✅ Using <header> tag")
            
            if 'class="post-header"' in tag_attrs:
                print("   ✅ Has class='post-header'")
            else:
                print("   ❌ Missing class='post-header'")
            
            # Check if tags are inside or outside post-meta
            header_inner = header_match.group(3)
            
            if '<div class="tags">' in header_inner:
                # Check if tags are inside post-meta
                meta_match = re.search(r'<div class="post-meta">(.*?)</div>', header_inner, re.DOTALL)
                if meta_match and '<div class="tags">' in meta_match.group(1):
                    print("\n   ❌ PROBLEM FOUND: Tags are INSIDE post-meta div")
                    print("      This causes left-alignment due to 'justify-content: space-between'")
                    print()
                    print("   📝 Structure should be:")
                    print("      <header class='post-header'>")
                    print("        <h1>Title</h1>")
                    print("        <div class='post-meta'><time>Date</time></div>")
                    print("        <div class='tags'>Tags here</div>  ← OUTSIDE post-meta")
                    print("      </header>")
                else:
                    print("\n   ✅ Tags are OUTSIDE post-meta (correct)")

print()
print("=" * 80)
print("SUMMARY & RECOMMENDED ACTIONS")
print("=" * 80)
print()

# Determine the main issues
issues = []
fixes = []

if not enhanced_css.exists():
    issues.append("enhanced-blog-post-styles.css not deployed to docs/static/")
    fixes.append("Copy static/enhanced-blog-post-styles.css to docs/static/")

if style_css.exists():
    with open(style_css, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if content.count('{') != content.count('}'):
        issues.append("CSS syntax error (mismatched braces)")
        fixes.append("Fix CSS syntax in style.css")
    
    if content.count("ENHANCED BLOG POST STYLING") > 1:
        issues.append("Duplicate enhanced blog post styles in style.css")
        fixes.append("Remove duplicate sections from style.css")

if html_file.exists():
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    
    if not any('enhanced-blog-post-styles.css' in link for link in re.findall(r'<link[^>]*href="([^"]*\.css)"', html)):
        issues.append("HTML not linking to enhanced-blog-post-styles.css")
        fixes.append("Add <link> tag to post template for enhanced-blog-post-styles.css")
    
    article_match = re.search(r'<article class="blog-post">(.*?)</article>', html, re.DOTALL)
    if article_match:
        article_content = article_match.group(1)
        header_match = re.search(r'<(header|div)([^>]*)>(.*?)</(header|div)>', article_content, re.DOTALL)
        if header_match:
            if 'class="post-header"' not in header_match.group(2):
                issues.append("Header missing class='post-header'")
                fixes.append("Run fix_generator.py or final_fix.py to fix template")
            
            meta_match = re.search(r'<div class="post-meta">(.*?)</div>', header_match.group(3), re.DOTALL)
            if meta_match and '<div class="tags">' in meta_match.group(1):
                issues.append("Tags are inside post-meta (causes left-alignment)")
                fixes.append("Run final_fix.py to move tags outside post-meta")

if issues:
    print("🔴 ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print()
    print("✅ RECOMMENDED FIXES:")
    for i, fix in enumerate(fixes, 1):
        print(f"   {i}. {fix}")
else:
    print("✅ No major issues detected!")
    print("   The CSS should be working. Try hard refresh (Ctrl+Shift+R)")

print()
print("=" * 80)
