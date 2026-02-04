#!/usr/bin/env python3
"""
Fix Post Header Overlapping Navigation Menu
Updates CSS files to prevent the gradient header from covering the nav
"""

from pathlib import Path
import re

print("=" * 80)
print("🔧 FIXING POST HEADER OVERLAP ISSUE")
print("=" * 80)
print()

# Files to update
files_to_update = [
    './docs/static/style.css',
    './static/style.css',
    './docs/static/enhanced-blog-post-styles.css',
    './static/enhanced-blog-post-styles.css'
]

def fix_header_margin(content):
    """Replace negative top margin with 0 in .post-header rules"""
    
    # Pattern 1: Find "margin: -2rem -2rem 3rem -2rem"
    pattern1 = r'margin:\s*-2rem\s+-2rem\s+3rem\s+-2rem'
    replacement1 = 'margin: 0 -2rem 3rem -2rem'
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: Find "margin: -2rem -1rem 2rem -1rem" (mobile)
    pattern2 = r'margin:\s*-2rem\s+-1rem\s+2rem\s+-1rem'
    replacement2 = 'margin: 0 -1rem 2rem -1rem'
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: Find "margin: -1rem -1rem 1.5rem -1rem" (small mobile)
    pattern3 = r'margin:\s*-1rem\s+-1rem\s+1\.5rem\s+-1rem'
    replacement3 = 'margin: 0 -1rem 1.5rem -1rem'
    content = re.sub(pattern3, replacement3, content)
    
    return content

fixed_count = 0

for file_path_str in files_to_update:
    file_path = Path(file_path_str)
    
    if not file_path.exists():
        continue
    
    print(f"📄 Processing: {file_path}")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Apply fix
    fixed_content = fix_header_margin(original_content)
    
    # Check if changes were made
    if fixed_content != original_content:
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"   ✅ Fixed negative margins in .post-header")
        fixed_count += 1
    else:
        print(f"   ℹ️  No changes needed")
    
    print()

print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print(f"Files updated: {fixed_count}")
print()

if fixed_count > 0:
    print("✅ SUCCESS! Fixed the header overlap issue")
    print()
    print("📝 WHAT WAS CHANGED:")
    print("   Changed: margin: -2rem -2rem 3rem -2rem;")
    print("   To:      margin: 0 -2rem 3rem -2rem;")
    print()
    print("   This removes the negative TOP margin that was pulling")
    print("   the header up and causing it to overlap the navigation menu.")
    print()
    print("=" * 80)
    print("🚀 NEXT STEPS:")
    print("=" * 80)
    print("1. Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)")
    print("2. If deploying: git add docs/static/*.css")
    print("3. Commit: git commit -m 'Fix: Post header no longer overlaps navigation'")
    print("4. Push: git push")
    print()
    print("The header will now sit below the sticky navigation menu! ✨")
else:
    print("ℹ️  No files were updated. The CSS might already be fixed,")
    print("   or the files weren't found in the expected locations.")

print()
print("=" * 80)
