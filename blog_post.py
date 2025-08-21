import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class BlogPost:
    def __init__(self, title, content, slug, tags, meta_description, featured_image,
                 created_at, updated_at, seo_keywords, affiliate_links=None, monetization_data=None):
        self.title = title
        self.content = content
        self.slug = slug
        self.tags = tags or []
        self.meta_description = meta_description
        self.featured_image = featured_image
        self.created_at = created_at
        self.updated_at = updated_at
        self.seo_keywords = seo_keywords or []
        self.affiliate_links = affiliate_links or []
        self.monetization_data = monetization_data or {}

    def to_dict(self):
        return {
            'title': self.title,
            'content': self.content,
            'slug': self.slug,
            'tags': self.tags,
            'meta_description': self.meta_description,
            'featured_image': self.featured_image,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'seo_keywords': self.seo_keywords,
            'affiliate_links': self.affiliate_links,
            'monetization_data': self.monetization_data
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_markdown_file(cls, md_file_path: Path, slug: str = None) -> 'BlogPost':
        """Create a BlogPost from a markdown file when post.json is missing"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        title = "Untitled Post"
        content_without_title = content
        
        if lines and lines[0].startswith('# '):
            title = lines[0][2:].strip()
            content_without_title = '\n'.join(lines[1:]).strip()
        
        if not slug:
            slug = cls._create_slug_static(title)
        
        current_time = datetime.now().isoformat()
        
        return cls(
            title=title,
            content=content_without_title,
            slug=slug,
            tags=['recovered', 'blog'],
            meta_description=f"Blog post about {title}",
            featured_image=f"/static/images/{slug}.jpg",
            created_at=current_time,
            updated_at=current_time,
            seo_keywords=[],
            affiliate_links=[],
            monetization_data={"ad_slots": 3, "affiliate_count": 0}
        )
    
    @staticmethod
    def _create_slug_static(title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]