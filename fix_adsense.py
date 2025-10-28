#!/usr/bin/env python3
"""
Complete fix for AdSense setup issues
Run this to fix all deployment problems
"""

import os
import shutil
from pathlib import Path

def fix_robots_txt():
    """Create correct robots.txt in docs folder"""
    print("1. Fixing robots.txt...")
    
    robots_content = """# Allow all crawlers access to all content
User-agent: *
Allow: /

# Specifically allow Google AdSense crawler (required for AdSense)
User-agent: Mediapartners-Google
Allow: /

# Allow Google's main crawler
User-agent: Googlebot
Allow: /

# Allow Google AdSense crawler
User-agent: Googlebot-Image
Allow: /

# Block crawlers from sensitive directories
User-agent: *
Disallow: /admin/
Disallow: /.git/

# IMPORTANT: Allow ads.txt
Allow: /ads.txt

# Sitemap location (CORRECTED URL)
Sitemap: https://kubaik.github.io/sitemap.xml

# Crawl delay (optional - seconds between requests)
Crawl-delay: 1
"""
    
    # Write to docs folder (GitHub Pages serves from here)
    with open('./docs/robots.txt', 'w', encoding='utf-8') as f:
        f.write(robots_content)
    
    print("   ✓ Created ./docs/robots.txt with correct sitemap URL")

def copy_ads_txt():
    """Ensure ads.txt is in docs folder"""
    print("\n2. Copying ads.txt to docs folder...")
    
    if os.path.exists('ads.txt'):
        shutil.copy('ads.txt', './docs/ads.txt')
        print("   ✓ Copied ads.txt to ./docs/")
    else:
        # Create it if it doesn't exist
        with open('./docs/ads.txt', 'w', encoding='utf-8') as f:
            f.write("google.com, pub-4477679588953789, DIRECT, f08c47fec0942fa0\n")
        print("   ✓ Created ./docs/ads.txt")

def verify_deployment_files():
    """Verify all required files exist in docs folder"""
    print("\n3. Verifying deployment files...")
    
    required_files = {
        './docs/ads.txt': 'AdSense ads.txt file',
        './docs/robots.txt': 'Robots.txt file',
        './docs/sitemap.xml': 'XML sitemap',
        './docs/index.html': 'Homepage',
        './docs/privacy-policy/index.html': 'Privacy Policy',
        './docs/terms-of-service/index.html': 'Terms of Service',
        './docs/about/index.html': 'About page',
        './docs/contact/index.html': 'Contact page'
    }
    
    missing = []
    present = []
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            present.append(f"   ✓ {description}: {file_path}")
        else:
            missing.append(f"   ✗ {description}: {file_path} - MISSING")
    
    print("\n   Present files:")
    for item in present:
        print(item)
    
    if missing:
        print("\n   Missing files:")
        for item in missing:
            print(item)
        return False
    
    return True

def check_file_contents():
    """Verify file contents are correct"""
    print("\n4. Checking file contents...")
    
    # Check ads.txt
    if os.path.exists('./docs/ads.txt'):
        with open('./docs/ads.txt', 'r') as f:
            content = f.read()
            if 'pub-4477679588953789' in content and 'f08c47fec0942fa0' in content:
                print("   ✓ ads.txt has correct publisher ID and relationship")
            else:
                print("   ✗ ads.txt content is incorrect")
    
    # Check robots.txt
    if os.path.exists('./docs/robots.txt'):
        with open('./docs/robots.txt', 'r') as f:
            content = f.read()
            if 'Mediapartners-Google' in content:
                print("   ✓ robots.txt allows AdSense crawler")
            else:
                print("   ✗ robots.txt missing AdSense crawler permission")
            
            if 'kubaik.github.io/sitemap.xml' in content:
                print("   ✓ robots.txt has correct sitemap URL")
            else:
                print("   ✗ robots.txt has wrong sitemap URL")

def count_blog_posts():
    """Count the number of blog posts"""
    print("\n5. Counting blog posts...")
    
    docs_path = Path('./docs')
    if not docs_path.exists():
        print("   ✗ docs directory doesn't exist")
        return 0
    
    post_count = 0
    for item in docs_path.iterdir():
        if item.is_dir() and (item / 'post.json').exists():
            post_count += 1
    
    print(f"   • Total blog posts: {post_count}")
    
    if post_count < 20:
        print(f"   ⚠ AdSense recommends at least 20-30 posts (you have {post_count})")
        print(f"   → Run this to generate more: python blog_system.py auto")
    else:
        print(f"   ✓ Good number of posts for AdSense approval")
    
    return post_count

def generate_deployment_checklist(post_count):
    """Generate a deployment checklist"""
    print("\n" + "="*60)
    print("DEPLOYMENT CHECKLIST")
    print("="*60)
    
    checklist = [
        ("ads.txt in docs folder", os.path.exists('./docs/ads.txt')),
        ("robots.txt in docs folder", os.path.exists('./docs/robots.txt')),
        ("sitemap.xml generated", os.path.exists('./docs/sitemap.xml')),
        ("Privacy Policy page", os.path.exists('./docs/privacy-policy/index.html')),
        ("Terms of Service page", os.path.exists('./docs/terms-of-service/index.html')),
        ("At least 20 posts", post_count >= 20)
    ]
    
    all_good = True
    for item, status in checklist:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {item}")
        if not status:
            all_good = False
    
    print("\n" + "="*60)
    
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to deploy!")
    else:
        print("⚠️  Some issues need attention before deployment")
    
    return all_good

def print_next_steps(ready_to_deploy):
    """Print next steps"""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if ready_to_deploy:
        print("""
1. Commit and push to GitHub:
   git add docs/
   git commit -m "Fix AdSense setup: correct robots.txt and ads.txt"
   git push origin main

2. Wait 24-48 hours for Google to crawl your site

3. Verify files are accessible:
   - https://kubaik.github.io/ads.txt
   - https://kubaik.github.io/robots.txt
   - https://kubaik.github.io/sitemap.xml

4. Check AdSense console after 48 hours

5. If "Needs attention" persists:
   - Generate more content (aim for 30+ posts)
   - Drive organic traffic to your site
   - Wait 2-4 weeks before reapplying
""")
    else:
        print("""
1. Fix missing issues first:
   python blog_system.py build

2. Generate more content if needed:
   for i in {1..10}; do python blog_system.py auto && sleep 5; done

3. Run this script again:
   python fix_adsense.py

4. Then commit and push to GitHub
""")

def main():
    print("="*60)
    print("ADSENSE FIX SCRIPT")
    print("="*60)
    print()
    
    # Run all fixes
    fix_robots_txt()
    copy_ads_txt()
    
    # Verify everything
    files_ok = verify_deployment_files()
    check_file_contents()
    post_count = count_blog_posts()
    
    # Generate checklist
    ready = generate_deployment_checklist(post_count)
    
    # Print next steps
    print_next_steps(ready)

if __name__ == "__main__":
    main()