import requests
import yaml
from urllib.parse import urlparse, parse_qs
import time
import logging

class AmazonAffiliateValidator:
    def __init__(self, config_path='config.yaml'):
        """Initialize with configuration from YAML files"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.affiliate_tag = self.config.get('amazon_affiliate_tag', 'aiblogcontent-20')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_asin(self, asin, marketplace='com'):
        """
        Validate if an ASIN exists on Amazon
        
        Args:
            asin (str): Amazon Standard Identification Number
            marketplace (str): Amazon marketplace (com, co.uk, de, etc.)
        
        Returns:
            dict: Validation result with status and details
        """
        base_urls = {
            'com': 'https://www.amazon.com',
            'co.uk': 'https://www.amazon.co.uk',
            'de': 'https://www.amazon.de',
            'fr': 'https://www.amazon.fr',
            'it': 'https://www.amazon.it',
            'es': 'https://www.amazon.es',
            'ca': 'https://www.amazon.ca',
            'au': 'https://www.amazon.com.au'
        }
        
        if marketplace not in base_urls:
            return {
                'valid': False,
                'error': f'Unsupported marketplace: {marketplace}',
                'status_code': None,
                'url': None
            }
        
        # Test both product URL formats
        urls_to_test = [
            f"{base_urls[marketplace]}/dp/{asin}",
            f"{base_urls[marketplace]}/gp/product/{asin}",
            f"{base_urls[marketplace]}/exec/obidos/ASIN/{asin}"
        ]
        
        for url in urls_to_test:
            try:
                self.logger.info(f"Testing URL: {url}")
                response = self.session.head(url, timeout=10, allow_redirects=True)
                
                if response.status_code == 200:
                    return {
                        'valid': True,
                        'status_code': response.status_code,
                        'url': url,
                        'final_url': response.url,
                        'marketplace': marketplace
                    }
                elif response.status_code == 404:
                    self.logger.warning(f"404 Not Found for {url}")
                else:
                    self.logger.warning(f"Status {response.status_code} for {url}")
                    
                time.sleep(1)  # Rate limiting
                
            except requests.RequestException as e:
                self.logger.error(f"Request failed for {url}: {str(e)}")
                continue
        
        return {
            'valid': False,
            'error': 'Product not found in any tested URL format',
            'status_code': 404,
            'marketplace': marketplace
        }
    
    def generate_affiliate_link(self, asin, marketplace='com', link_format='short'):
        """
        Generate affiliate link for a valid ASIN
        
        Args:
            asin (str): Amazon Standard Identification Number
            marketplace (str): Amazon marketplace
            link_format (str): 'short' or 'long' format
        
        Returns:
            str: Affiliate link or None if invalid
        """
        validation = self.validate_asin(asin, marketplace)
        
        if not validation['valid']:
            self.logger.error(f"Cannot generate link for invalid ASIN: {asin}")
            return None
        
        base_urls = {
            'com': 'https://www.amazon.com',
            'co.uk': 'https://www.amazon.co.uk',
            'de': 'https://www.amazon.de',
            'fr': 'https://www.amazon.fr',
            'it': 'https://www.amazon.it',
            'es': 'https://www.amazon.es',
            'ca': 'https://www.amazon.ca',
            'au': 'https://www.amazon.com.au'
        }
        
        base_url = base_urls[marketplace]
        
        if link_format == 'short':
            return f"{base_url}/dp/{asin}?tag={self.affiliate_tag}"
        else:
            return f"{base_url}/gp/product/{asin}?ie=UTF8&tag={self.affiliate_tag}"
    
    def test_current_links(self, links):
        """
        Test a list of current affiliate links
        
        Args:
            links (list): List of affiliate links to test
        
        Returns:
            list: Results for each link
        """
        results = []
        
        for link in links:
            try:
                # Parse ASIN from link
                parsed_url = urlparse(link)
                path_parts = parsed_url.path.strip('/').split('/')
                
                asin = None
                if 'dp' in path_parts:
                    dp_index = path_parts.index('dp')
                    if dp_index + 1 < len(path_parts):
                        asin = path_parts[dp_index + 1]
                elif 'product' in path_parts:
                    product_index = path_parts.index('product')
                    if product_index + 1 < len(path_parts):
                        asin = path_parts[product_index + 1]
                
                if not asin:
                    results.append({
                        'link': link,
                        'valid': False,
                        'error': 'Could not extract ASIN from URL'
                    })
                    continue
                
                # Determine marketplace
                domain = parsed_url.netloc.lower()
                marketplace = 'com'  # default
                if 'amazon.co.uk' in domain:
                    marketplace = 'co.uk'
                elif 'amazon.de' in domain:
                    marketplace = 'de'
                elif 'amazon.fr' in domain:
                    marketplace = 'fr'
                # Add more as needed
                
                # Validate the ASIN
                validation = self.validate_asin(asin, marketplace)
                validation['link'] = link
                validation['asin'] = asin
                results.append(validation)
                
            except Exception as e:
                results.append({
                    'link': link,
                    'valid': False,
                    'error': f'Error processing link: {str(e)}'
                })
        
        return results
    
    def find_alternative_products(self, category, marketplace='com'):
        """
        This would integrate with Amazon API to find alternative products
        For now, returns popular ASINs by category as examples
        """
        # Sample popular products by category
        sample_products = {
            'tech': [
                'B08N5WRWNW',  # Echo Dot 4th Gen
                'B0BXZP9G9Y',  # iPad 10th Gen
                'B09G9FPHY6'   # MacBook Air M1
            ],
            'books': [
                'B073JBQZPX',  # Programming books
                'B08CKQZGBW',  # AI/ML books
                'B07XJ8C8F7'   # Development books
            ],
            'electronics': [
                'B08P2MJWNZ',  # Bluetooth headphones
                'B09D3D5547',  # Webcam
                'B08GYM5F8G'   # USB-C Hub
            ]
        }
        
        category_products = sample_products.get(category.lower(), sample_products['tech'])
        
        valid_products = []
        for asin in category_products:
            validation = self.validate_asin(asin, marketplace)
            if validation['valid']:
                affiliate_link = self.generate_affiliate_link(asin, marketplace)
                valid_products.append({
                    'asin': asin,
                    'affiliate_link': affiliate_link,
                    'validation': validation
                })
        
        return valid_products


# Example usage and testing
if __name__ == "__main__":
    validator = AmazonAffiliateValidator()
    
    # Test the problematic ASIN
    print("=== Testing Problematic ASIN ===")
    problem_asin = "B07C3KLQWX"
    result = validator.validate_asin(problem_asin)
    print(f"ASIN {problem_asin}: {result}")
    
    # Test the full problematic link
    print("\n=== Testing Full Link ===")
    problem_link = "https://www.amazon.com/dp/B07C3KLQWX?tag=aiblogcontent-20"
    link_results = validator.test_current_links([problem_link])
    print(f"Link test results: {link_results[0]}")
    
    # Find alternative products
    print("\n=== Finding Alternative Products ===")
    alternatives = validator.find_alternative_products('tech')
    print("Valid alternative products:")
    for product in alternatives[:3]:  # Show first 3
        print(f"  ASIN: {product['asin']}")
        print(f"  Link: {product['affiliate_link']}")
        print()
    
    # Test multiple marketplaces
    print("=== Testing Multiple Marketplaces ===")
    test_asin = "B08N5WRWNW"  # Echo Dot 4th Gen
    marketplaces = ['com', 'co.uk', 'de']
    
    for marketplace in marketplaces:
        result = validator.validate_asin(test_asin, marketplace)
        if result['valid']:
            link = validator.generate_affiliate_link(test_asin, marketplace)
            print(f"  {marketplace}: {link}")
        else:
            print(f"  {marketplace}: Not available")
