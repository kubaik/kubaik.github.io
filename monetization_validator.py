import yaml
import requests
import json
from urllib.parse import urlparse, parse_qs
import logging

class MonetizationDebugger:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.affiliate_tag = self.config.get('amazon_affiliate_tag', 'aiblogcontent-20')
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Different user agents to test
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    def debug_affiliate_link(self, asin, full_debug=True):
        """Debug affiliate link generation and loading"""
        print(f"\n{'='*60}")
        print(f"DEBUGGING AFFILIATE LINK FOR ASIN: {asin}")
        print(f"{'='*60}")
        
        # 1. Test basic product URL without affiliate tag
        base_url = f"https://www.amazon.com/dp/{asin}"
        print(f"\n1. Testing base product URL: {base_url}")
        self._test_url(base_url)
        
        # 2. Test with affiliate tag
        affiliate_url = f"{base_url}?tag={self.affiliate_tag}"
        print(f"\n2. Testing affiliate URL: {affiliate_url}")
        self._test_url(affiliate_url)
        
        # 3. Test different URL formats
        print(f"\n3. Testing different URL formats:")
        url_formats = [
            f"https://www.amazon.com/dp/{asin}?tag={self.affiliate_tag}",
            f"https://www.amazon.com/gp/product/{asin}?tag={self.affiliate_tag}",
            f"https://www.amazon.com/exec/obidos/ASIN/{asin}?tag={self.affiliate_tag}",
            f"https://amzn.to/{asin}?tag={self.affiliate_tag}"  # Short URL format
        ]
        
        for url in url_formats:
            print(f"  Testing: {url}")
            result = self._test_url(url, verbose=False)
            print(f"    Result: {result['status']} - {result.get('final_url', 'N/A')}")
        
        # 4. Test with different user agents
        print(f"\n4. Testing with different User-Agents:")
        for i, ua in enumerate(self.user_agents, 1):
            print(f"  User-Agent {i}: {ua[:50]}...")
            result = self._test_url_with_ua(affiliate_url, ua)
            print(f"    Result: {result['status']} - {result.get('redirect_count', 0)} redirects")
        
        # 5. Test affiliate tag validation
        print(f"\n5. Validating affiliate tag: {self.affiliate_tag}")
        self._validate_affiliate_tag()
        
        # 6. Test CORS and referrer policies
        print(f"\n6. Testing CORS and referrer policies:")
        self._test_cors_referrer(affiliate_url)
        
        if full_debug:
            # 7. Analyze response headers
            print(f"\n7. Analyzing response headers:")
            self._analyze_headers(affiliate_url)
    
    def _test_url(self, url, verbose=True):
        """Test URL with detailed response analysis"""
        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': self.user_agents[0],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            response = session.get(url, timeout=15, allow_redirects=True)
            
            result = {
                'status': f"{response.status_code} {response.reason}",
                'final_url': response.url,
                'redirect_count': len(response.history),
                'content_type': response.headers.get('content-type', 'Unknown'),
                'content_length': response.headers.get('content-length', 'Unknown')
            }
            
            if verbose:
                print(f"    Status: {result['status']}")
                print(f"    Final URL: {result['final_url']}")
                print(f"    Redirects: {result['redirect_count']}")
                print(f"    Content-Type: {result['content_type']}")
                print(f"    Content-Length: {result['content_length']}")
                
                if response.history:
                    print(f"    Redirect chain:")
                    for i, redirect in enumerate(response.history, 1):
                        print(f"      {i}. {redirect.status_code} -> {redirect.url}")
            
            # Check if it's actually an Amazon product page
            if response.status_code == 200:
                content_snippet = response.text[:1000].lower()
                if 'amazon' not in content_snippet:
                    print(f"    WARNING: Response doesn't appear to be from Amazon")
                    result['warning'] = 'Not Amazon content'
            
            return result
            
        except requests.exceptions.RequestException as e:
            error_result = {'status': f'ERROR: {str(e)}', 'error': True}
            if verbose:
                print(f"    ERROR: {str(e)}")
            return error_result
    
    def _test_url_with_ua(self, url, user_agent):
        """Test URL with specific user agent"""
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': user_agent})
            
            response = session.get(url, timeout=10, allow_redirects=True)
            return {
                'status': f"{response.status_code} {response.reason}",
                'redirect_count': len(response.history)
            }
        except requests.exceptions.RequestException as e:
            return {'status': f'ERROR: {str(e)}', 'error': True}
    
    def _validate_affiliate_tag(self):
        """Validate affiliate tag format"""
        tag = self.affiliate_tag
        
        print(f"    Tag: '{tag}'")
        print(f"    Length: {len(tag)} characters")
        
        # Amazon affiliate tag rules
        if len(tag) > 20:
            print(f"    WARNING: Tag is longer than 20 characters")
        
        if not tag.replace('-', '').replace('_', '').isalnum():
            print(f"    WARNING: Tag contains special characters other than - and _")
        
        if tag.endswith('-20'):
            print(f"    INFO: Tag ends with '-20' (common Amazon affiliate format)")
        else:
            print(f"    INFO: Tag doesn't end with '-20'")
    
    def _test_cors_referrer(self, url):
        """Test CORS and referrer policies"""
        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': self.user_agents[0],
                'Referer': self.config.get('base_url', 'https://localhost:3000'),
                'Origin': self.config.get('base_url', 'https://localhost:3000')
            })
            
            response = session.head(url, timeout=10, allow_redirects=True)
            
            print(f"    Status with referrer: {response.status_code}")
            print(f"    CORS headers present: {'access-control-allow-origin' in response.headers}")
            
            referrer_policy = response.headers.get('referrer-policy', 'Not set')
            print(f"    Referrer Policy: {referrer_policy}")
            
        except requests.exceptions.RequestException as e:
            print(f"    ERROR testing CORS/Referrer: {str(e)}")
    
    def _analyze_headers(self, url):
        """Analyze response headers for debugging"""
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': self.user_agents[0]})
            
            response = session.head(url, timeout=10, allow_redirects=True)
            
            important_headers = [
                'cache-control', 'expires', 'last-modified', 'etag',
                'x-frame-options', 'content-security-policy',
                'strict-transport-security', 'x-content-type-options'
            ]
            
            print(f"    Important headers:")
            for header in important_headers:
                value = response.headers.get(header, 'Not present')
                print(f"      {header}: {value}")
            
        except requests.exceptions.RequestException as e:
            print(f"    ERROR analyzing headers: {str(e)}")
    
    def check_website_integration(self):
        """Check how affiliate links should be integrated in website"""
        print(f"\n{'='*60}")
        print("WEBSITE INTEGRATION CHECKLIST")
        print(f"{'='*60}")
        
        base_url = self.config.get('base_url', 'Not configured')
        print(f"Website base URL: {base_url}")
        
        print(f"\n1. HTML Integration:")
        sample_html = f'''<a href="https://www.amazon.com/dp/B07C3KLQWX?tag={self.affiliate_tag}" 
   target="_blank" 
   rel="noopener nofollow">
   Check out this product on Amazon
</a>'''
        print(sample_html)
        
        print(f"\n2. JavaScript Integration:")
        sample_js = f'''// Proper way to handle affiliate links in JavaScript
const affiliateLink = "https://www.amazon.com/dp/B07C3KLQWX?tag={self.affiliate_tag}";

// Open in new tab
function openAffiliateLink() {{
    window.open(affiliateLink, '_blank', 'noopener,noreferrer');
}}

// Or redirect current page
function redirectToAffiliate() {{
    window.location.href = affiliateLink;
}}'''
        print(sample_js)
        
        print(f"\n3. Common Issues to Check:")
        issues = [
            "• Ensure links open in new tab/window (target='_blank')",
            "• Add rel='nofollow noopener' for SEO and security",
            "• Check if ad blockers are interfering",
            "• Verify affiliate tag is correctly formatted",
            "• Ensure HTTPS is used (not HTTP)",
            "• Check if CSP (Content Security Policy) blocks external links",
            "• Test on different browsers and devices"
        ]
        for issue in issues:
            print(issue)
    
    def generate_working_links(self, asins):
        """Generate properly formatted working affiliate links"""
        print(f"\n{'='*60}")
        print("WORKING AFFILIATE LINKS")
        print(f"{'='*60}")
        
        for asin in asins:
            # Test the ASIN first
            base_url = f"https://www.amazon.com/dp/{asin}"
            result = self._test_url(base_url, verbose=False)
            
            if 'ERROR' not in result['status'] and '200' in result['status']:
                affiliate_link = f"{base_url}?tag={self.affiliate_tag}"
                print(f"\nASIN: {asin}")
                print(f"Working link: {affiliate_link}")
                
                # Test the affiliate link too
                affiliate_result = self._test_url(affiliate_link, verbose=False)
                if 'ERROR' not in affiliate_result['status']:
                    print(f"Status: ✓ {affiliate_result['status']}")
                else:
                    print(f"Status: ✗ {affiliate_result['status']}")
            else:
                print(f"\nASIN: {asin} - NOT WORKING ({result['status']})")


# Usage
if __name__ == "__main__":
    debugger = MonetizationDebugger()
    
    # Debug the problematic ASIN
    debugger.debug_affiliate_link("B07C3KLQWX")
    
    # Check website integration
    debugger.check_website_integration()
    
    # Generate working links for backup ASINs
    backup_asins = ["B08N5WRWNW", "B0BXZP9G9Y", "B09G9FPHY6"]
    debugger.generate_working_links(backup_asins)