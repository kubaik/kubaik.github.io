#!/usr/bin/env python3
"""
Complete blog diagnostic and fix tool
Analyzes current state and provides actionable fixes
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

class BlogDiagnostic:
    def __init__(self):
        self.docs_path = Path('./docs')
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def check_directory_structure(self):
        """Check if basic directory structure exists"""
        print("="*60)
        print("CHECKING DIRECTORY STRUCTURE")
        print("="*60)
        
        if not self.docs_path.exists():
            self.issues.append("docs/ directory doesn't exist")
            print("✗ docs/ directory not found")
            print("  → Run: mkdir -p docs/static")
            return False
        else:
            self.successes.append("docs/ directory exists")
            print("✓ docs/ directory exists")
        
        static_path = self.docs_path / 'static'
        if not static_path.exists():
            self.warnings.append("static/ directory missing")
            print("✗ docs/static/ directory not found")
            print("  → Run: mkdir -p docs/static")
        else:
            self.successes.append("static/ directory exists")
            print("✓ docs/static/ directory exists")
        
        return True
    
    def count_posts(self):
        """Count blog posts and analyze them"""
        print("\n" + "="*60)
        print("ANALYZING BLOG POSTS")
        print("="*60)
        
        if not self.docs_path.exists():
            print("✗ Cannot count posts - docs/ doesn't exist")
            return 0
        
        posts = []
        total_words = 0
        
        for item in self.docs_path.iterdir():
            if not item.is_dir() or item.name == 'static':
                continue
            
            post_json = item / 'post.json'
            if post_json.exists():
                try:
                    with open(post_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Count words in content
                    word_count = len(data.get('content', '').split())
                    total_words += word_count
                    
                    posts.append({
                        'slug': item.name,
                        'title': data.get('title', 'Unknown'),
                        'word_count': word_count,
                        'created': data.get('created_at', 'Unknown'),
                        'has_seo': bool(data.get('seo_keywords')),
                        'has_meta': bool(data.get('meta_description')),
                        'affiliate_links': len(data.get('affiliate_links', [])),
                        'ad_slots': data.get('monetization_data', {}).get('ad_slots', 0)
                    })
                except Exception as e:
                    self.warnings.append(f"Error reading {item.name}: {e}")
        
        # Sort by creation date
        posts.sort(key=lambda x: x['created'], reverse=True)
        
        print(f"\n📊 BLOG STATISTICS:")
        print(f"   • Total posts: {len(posts)}")
        print(f"   • Total words: {total_words:,}")
        print(f"   • Average words/post: {total_words // len(posts) if posts else 0}")
        
        if len(posts) < 20:
            self.issues.append(f"Only {len(posts)} posts (need 20-30 minimum)")
            print(f"\n⚠️  You have {len(posts)} posts")
            print(f"   AdSense recommends 20-30 quality posts minimum")
            print(f"   → Need {20 - len(posts)} more posts")
        elif len(posts) < 30:
            self.warnings.append(f"Only {len(posts)} posts (30+ recommended)")
            print(f"\n✓ You have {len(posts)} posts (good, but 30+ is better)")
        else:
            self.successes.append(f"Excellent post count: {len(posts)}")
            print(f"\n✓ Excellent! You have {len(posts)} posts")
        
        # Show recent posts
        print(f"\n📝 RECENT POSTS (last 5):")
        for i, post in enumerate(posts[:5], 1):
            status = "✓" if post['word_count'] >= 800 else "⚠"
            print(f"   {status} {post['title'][:50]}")
            print(f"      Words: {post['word_count']}, SEO: {'Yes' if post['has_seo'] else 'No'}, "
                  f"Ads: {post['ad_slots']}")
        
        # Check content quality
        short_posts = [p for p in posts if p['word_count'] < 800]
        if short_posts:
            self.warnings.append(f"{len(short_posts)} posts under 800 words")
            print(f"\n⚠️  {len(short_posts)} posts are under 800 words")
            print(f"   AdSense prefers longer, quality content")
        
        return len(posts)
    
    def check_required_files(self):
        """Check all required files for AdSense"""
        print("\n" + "="*60)
        print("CHECKING REQUIRED FILES")
        print("="*60)
        
        files_to_check = {
            'ads.txt': './docs/ads.txt',
            'robots.txt': './docs/robots.txt',
            'sitemap.xml': './docs/sitemap.xml',
            'rss.xml': './docs/rss.xml',
            'index.html': './docs/index.html',
            'style.css': './docs/static/style.css'
        }
        
        for name, path in files_to_check.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                print(f"✓ {name:15} exists ({file_size:,} bytes)")
                self.successes.append(f"{name} exists")
                
                # Special checks for critical files
                if name == 'ads.txt':
                    with open(path, 'r') as f:
                        content = f.read()
                        if 'pub-4477679588953789' in content:
                            print(f"  → Publisher ID verified: pub-4477679588953789")
                        else:
                            self.issues.append("ads.txt has wrong publisher ID")
                            print(f"  ✗ Wrong publisher ID in ads.txt")
                
                elif name == 'robots.txt':
                    with open(path, 'r') as f:
                        content = f.read()
                        if 'Mediapartners-Google' in content:
                            print(f"  → AdSense crawler allowed ✓")
                        else:
                            self.issues.append("robots.txt doesn't allow AdSense crawler")
                            print(f"  ✗ Missing AdSense crawler permission")
                        
                        if 'kubaik.github.io/sitemap.xml' in content:
                            print(f"  → Sitemap URL correct ✓")
                        else:
                            self.issues.append("robots.txt has wrong sitemap URL")
                            print(f"  ✗ Wrong sitemap URL")
            else:
                self.issues.append(f"Missing {name}")
                print(f"✗ {name:15} MISSING")
                print(f"  → Run: python blog_system.py build")
    
    def check_required_pages(self):
        """Check required pages for AdSense compliance"""
        print("\n" + "="*60)
        print("CHECKING REQUIRED PAGES")
        print("="*60)
        
        pages = {
            'Privacy Policy': './docs/privacy-policy/index.html',
            'Terms of Service': './docs/terms-of-service/index.html',
            'About': './docs/about/index.html',
            'Contact': './docs/contact/index.html'
        }
        
        for name, path in pages.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                print(f"✓ {name:20} exists ({file_size:,} bytes)")
                self.successes.append(f"{name} page exists")
            else:
                self.issues.append(f"Missing {name} page")
                print(f"✗ {name:20} MISSING")
                print(f"  → Run: python blog_system.py build")
    
    def check_config(self):
        """Check configuration file"""
        print("\n" + "="*60)
        print("CHECKING CONFIGURATION")
        print("="*60)
        
        if not os.path.exists('config.yaml'):
            self.issues.append("config.yaml missing")
            print("✗ config.yaml not found")
            print("  → Run: python blog_system.py init")
            return
        
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical settings
        critical_settings = {
            'google_adsense_id': config.get('google_adsense_id'),
            'google_analytics_id': config.get('google_analytics_id'),
            'base_url': config.get('base_url'),
            'site_name': config.get('site_name')
        }
        
        for key, value in critical_settings.items():
            if value:
                print(f"✓ {key:25} {value}")
                self.successes.append(f"{key} configured")
            else:
                self.warnings.append(f"{key} not configured")
                print(f"⚠ {key:25} NOT SET")
    
    def fix_robots_txt(self):
        """Fix robots.txt with correct settings"""
        print("\n" + "="*60)
        print("FIXING ROBOTS.TXT")
        print("="*60)
        
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

# Sitemap location
Sitemap: https://kubaik.github.io/sitemap.xml

# Crawl delay
Crawl-delay: 1
"""
        
        try:
            with open('./docs/robots.txt', 'w', encoding='utf-8') as f:
                f.write(robots_content)
            print("✓ Created/updated robots.txt with correct settings")
            return True
        except Exception as e:
            print(f"✗ Failed to create robots.txt: {e}")
            return False
    
    def ensure_ads_txt(self):
        """Ensure ads.txt is in the right place"""
        print("\n" + "="*60)
        print("ENSURING ADS.TXT")
        print("="*60)
        
        ads_content = "google.com, pub-4477679588953789, DIRECT, f08c47fec0942fa0\n"
        
        try:
            with open('./docs/ads.txt', 'w', encoding='utf-8') as f:
                f.write(ads_content)
            print("✓ Created/updated ads.txt in docs folder")
            return True
        except Exception as e:
            print(f"✗ Failed to create ads.txt: {e}")
            return False
    
    def generate_report(self, post_count):
        """Generate final diagnostic report"""
        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)
        
        print(f"\n✅ SUCCESSES ({len(self.successes)}):")
        for success in self.successes[:10]:  # Show top 10
            print(f"   ✓ {success}")
        if len(self.successes) > 10:
            print(f"   ... and {len(self.successes) - 10} more")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠ {warning}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   ✗ {issue}")
        
        # Overall score
        total_checks = len(self.successes) + len(self.warnings) + len(self.issues)
        score = (len(self.successes) * 1.0 + len(self.warnings) * 0.5) / total_checks * 100 if total_checks > 0 else 0
        
        print(f"\n📊 OVERALL READINESS SCORE: {score:.1f}%")
        
        if score >= 90:
            print("   🎉 Excellent! Your blog is ready for AdSense")
        elif score >= 70:
            print("   👍 Good progress! Fix the issues and you'll be ready")
        elif score >= 50:
            print("   ⚠️  Needs work. Address the critical issues first")
        else:
            print("   ❌ Major issues need attention before AdSense approval")
        
        return len(self.issues) == 0
    
    def print_action_plan(self, post_count):
        """Print action plan based on findings"""
        print("\n" + "="*60)
        print("ACTION PLAN")
        print("="*60)
        
        actions = []
        
        if self.issues or self.warnings:
            actions.append(("HIGH", "Run: python blog_system.py build", 
                          "Rebuild site with all required files"))
        
        if post_count < 20:
            needed = 20 - post_count
            actions.append(("HIGH", f"Generate {needed} more posts",
                          f"Run: python blog_system.py auto (repeat {needed} times)"))
        elif post_count < 30:
            actions.append(("MEDIUM", "Generate more posts for safety",
                          "Aim for 30+ total posts"))
        
        if any('robots.txt' in str(issue) for issue in self.issues):
            actions.append(("HIGH", "Fix robots.txt", 
                          "Already fixed by this script - commit changes"))
        
        actions.append(("HIGH", "Commit and push to GitHub",
                       "git add docs/ && git commit -m 'Fix AdSense setup' && git push"))
        
        actions.append(("MEDIUM", "Wait for Google to re-crawl",
                       "Give it 24-48 hours after pushing"))
        
        actions.append(("MEDIUM", "Verify files are accessible",
                       "Check https://kubaik.github.io/ads.txt and other files"))
        
        actions.append(("LOW", "Drive traffic to your site",
                       "Share posts on social media, forums, etc."))
        
        # Print actions by priority
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            priority_actions = [a for a in actions if a[0] == priority]
            if priority_actions:
                print(f"\n{priority} PRIORITY:")
                for _, action, description in priority_actions:
                    print(f"  → {action}")
                    print(f"    {description}")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic"""
        print("\n" + "="*70)
        print("    BLOG ADSENSE DIAGNOSTIC TOOL")
        print("="*70)
        print(f"    Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Run all checks
        self.check_directory_structure()
        post_count = self.count_posts()
        self.check_required_files()
        self.check_required_pages()
        self.check_config()
        
        # Apply fixes
        self.fix_robots_txt()
        self.ensure_ads_txt()
        
        # Generate report
        ready = self.generate_report(post_count)
        self.print_action_plan(post_count)
        
        print("\n" + "="*70)
        print("    DIAGNOSTIC COMPLETE")
        print("="*70)
        
        return ready, post_count

def main():
    diagnostic = BlogDiagnostic()
    ready, post_count = diagnostic.run_full_diagnostic()
    
    print("\n💡 TIP: Run this script anytime to check your blog's status")
    print("   Save as: diagnostic.py")
    print()

if __name__ == "__main__":
    main()