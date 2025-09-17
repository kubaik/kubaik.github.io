import random
from typing import Dict, List, Tuple

class MonetizationManager:
    """Handles automated monetization features"""
    
    def __init__(self, config):
        self.config = config
        self.affiliate_programs = {
            'amazon': {
                'tag': config.get('amazon_affiliate_tag', 'aiblogcontent-20'),
                'api_key': config.get('amazon_api_key'),
                'keywords': ['tools', 'software', 'books', 'course', 'equipment']
            },
            'tech_products': {
                'categories': ['software', 'cloud', 'development', 'ai-tools'],
                'commission_rate': 0.05
            }
        }
        
    def inject_affiliate_links(self, content: str, topic: str) -> tuple:
        """Inject relevant affiliate links into content"""
        affiliate_links = []
        enhanced_content = content
        
        suggestions = self._get_affiliate_suggestions(topic)
        
        for suggestion in suggestions[:3]:  # Limit to 3 per post
            link_html = f'<a href="{suggestion["url"]}" target="_blank" rel="nofollow sponsored">{suggestion["text"]}</a>'
            
            insertion_points = self._find_insertion_points(content, suggestion["keywords"])
            
            if insertion_points:
                insert_at = random.choice(insertion_points)
                lines = enhanced_content.split('\n')
                if insert_at < len(lines):
                    lines[insert_at] += f"\n\n*Recommended: {link_html}*\n"
                enhanced_content = '\n'.join(lines)
                
                affiliate_links.append({
                    'url': suggestion['url'],
                    'text': suggestion['text'],
                    'commission_rate': suggestion.get('commission', 0.05)
                })
        
        return enhanced_content, affiliate_links
    
    def _get_affiliate_suggestions(self, topic: str) -> list:
        """Get relevant affiliate suggestions based on topic"""
        suggestions = []
        topic_lower = topic.lower()
        
        # AI/ML Tools
        if any(term in topic_lower for term in ['ai', 'machine learning', 'data science']):
            suggestions.extend([
                {
                    'url': f'https://amazon.com/dp/B08N5WRWNW?tag={self.affiliate_programs["amazon"]["tag"]}',
                    'text': 'Python Machine Learning by Sebastian Raschka',
                    'keywords': ['python', 'learning', 'algorithm'],
                    'commission': 0.04
                },
                {
                    'url': 'https://coursera.org/learn/machine-learning',
                    'text': 'Andrew Ng\'s Machine Learning Course',
                    'keywords': ['course', 'learn', 'training'],
                    'commission': 0.10
                }
            ])
        
        # Web Development
        if any(term in topic_lower for term in ['web', 'frontend', 'backend', 'javascript']):
            suggestions.extend([
                {
                    'url': f'https://amazon.com/dp/B07C3KLQWX?tag={self.affiliate_programs["amazon"]["tag"]}',
                    'text': 'Eloquent JavaScript Book',
                    'keywords': ['javascript', 'programming', 'web'],
                    'commission': 0.04
                },
                {
                    'url': 'https://digitalocean.com',
                    'text': 'DigitalOcean Cloud Hosting',
                    'keywords': ['hosting', 'deploy', 'server'],
                    'commission': 0.25
                }
            ])
            
        # DevOps/Cloud
        if any(term in topic_lower for term in ['devops', 'cloud', 'aws', 'docker']):
            suggestions.extend([
                {
                    'url': f'https://amazon.com/dp/B0816Q9F6Z?tag={self.affiliate_programs["amazon"]["tag"]}',
                    'text': 'Docker Deep Dive by Nigel Poulton',
                    'keywords': ['docker', 'container', 'devops'],
                    'commission': 0.04
                }
            ])
        
        return suggestions
    
    def _find_insertion_points(self, content: str, keywords: list) -> list:
        """Find good places to insert affiliate links"""
        lines = content.split('\n')
        insertion_points = []
        
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                insertion_points.append(i)
        
        return insertion_points
    
    def generate_ad_slots(self, content: str) -> dict:
        """Generate ad slot positions in content"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        ad_slots = {
            'header': 2,
            'middle': total_lines // 2,
            'footer': total_lines - 3,
            'ad_slots': 3,
            'affiliate_count': 0
        }
        
        return ad_slots