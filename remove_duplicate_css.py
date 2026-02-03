#!/usr/bin/env python3
"""
Remove duplicate enhanced blog post styles from style.css
Keeps only ONE copy of the enhanced styles
"""

from pathlib import Path
import re

print("=" * 80)
print("🧹 REMOVING DUPLICATE CSS FROM style.css")
print("=" * 80)
print()

style_path = Path('./static/style.css')

if not style_path.exists():
    print("❌ ERROR: static/style.css not found")
    print("   Make sure you're running this from your project root directory")
    exit(1)

# Read the file
with open(style_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("📊 Current file stats:")
print(f"   Total size: {len(content):,} characters")
print(f"   Lines: {content.count(chr(10)):,}")
print()

# Count how many times the enhanced blog post section appears
enhanced_count = content.count("ENHANCED BLOG POST STYLING")
print(f"📊 'ENHANCED BLOG POST STYLING' appears: {enhanced_count} times")

if enhanced_count <= 1:
    print()
    print("✅ No duplicates found! Your style.css is clean.")
    print()
    print("The issue might be elsewhere. Let me check the template structure...")
    print()
    print("Run this command to check your template:")
    print("   python css_diagnostic.py")
    exit(0)

print(f"   ⚠️  Found {enhanced_count} copies (should only have 1)")
print()

# Strategy: Keep everything up to and including the FIRST occurrence,
# then skip to the section after the SECOND occurrence

# Find the first occurrence
first_marker = "/* ============================================\n   ENHANCED BLOG POST STYLING"
first_pos = content.find(first_marker)

if first_pos == -1:
    print("❌ Could not find enhanced blog post styling marker")
    exit(1)

# Find the END of the first enhanced section
# It ends at "/* Tags (for other places like post cards) */"
end_marker = "/* Tags (for other places like post cards) */"
first_end_pos = content.find(end_marker, first_pos)

if first_end_pos == -1:
    print("❌ Could not find end marker for enhanced section")
    exit(1)

# Find where the SECOND enhanced section starts (if it exists)
second_pos = content.find(first_marker, first_pos + 1)

if second_pos == -1:
    print("✅ Only one enhanced section found. File is clean!")
    exit(0)

# Find the end of the SECOND enhanced section
second_end_pos = content.find(end_marker, second_pos)

if second_end_pos == -1:
    # If we can't find the second end, just cut from second start to the first end marker after it
    second_end_pos = content.find("\n/* ", second_pos + len(first_marker))

# Reconstruct the file:
# 1. Keep everything before the second enhanced section
# 2. Skip the second enhanced section
# 3. Keep everything after

clean_content = content[:second_pos] + content[second_end_pos:]

print("🔧 Removing duplicate section...")
print(f"   Keeping: Characters 0 to {second_pos:,}")
print(f"   Removing: Characters {second_pos:,} to {second_end_pos:,} (duplicate)")
print(f"   Keeping: Characters {second_end_pos:,} to end")
print()

# Verify the result
clean_enhanced_count = clean_content.count("ENHANCED BLOG POST STYLING")
print(f"📊 After cleanup:")
print(f"   'ENHANCED BLOG POST STYLING' appears: {clean_enhanced_count} times")
print(f"   Size reduced: {len(content):,} → {len(clean_content):,} characters")
print(f"   Lines reduced: {content.count(chr(10)):,} → {clean_content.count(chr(10)):,}")
print()

# Check CSS syntax
open_braces = clean_content.count('{')
close_braces = clean_content.count('}')
print(f"📊 CSS syntax check:")
print(f"   Open braces: {open_braces}")
print(f"   Close braces: {close_braces}")

if open_braces != close_braces:
    print(f"   ⚠️  WARNING: Mismatched braces! Difference: {open_braces - close_braces}")
    print("   The cleanup may have caused a syntax error.")
    print("   Creating backup before proceeding...")
    
    # Create backup
    backup_path = Path('./static/style.css.backup')
    import shutil
    shutil.copy(style_path, backup_path)
    print(f"   ✅ Backup saved to: {backup_path}")
else:
    print("   ✅ Braces are balanced")

print()

# Ask for confirmation
print("=" * 80)
response = input("❓ Save the cleaned version? (y/n): ").strip().lower()

if response == 'y':
    # Save the cleaned version
    with open(style_path, 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    print()
    print("✅ SUCCESS! style.css has been cleaned")
    print()
    print("=" * 80)
    print("🚀 NEXT STEPS:")
    print("=" * 80)
    print("1. Test locally: Open a blog post HTML file in your browser")
    print("2. If it looks good, deploy:")
    print("   git add static/style.css")
    print("   git commit -m 'Fix: Remove duplicate CSS sections'")
    print("   git push")
    print("3. Wait 2 minutes, then hard refresh (Ctrl+Shift+R)")
else:
    print()
    print("❌ Cleanup cancelled. No changes made.")
    print("   Review the stats above and run again when ready.")

print()
print("=" * 80)
