import re
from datetime import datetime
from pathlib import Path


class BlogPost:
    def __init__(
        self,
        title,
        content,
        slug,
        tags,
        meta_description,
        featured_image,
        created_at,
        updated_at,
        seo_keywords,
        affiliate_links=None,
        monetization_data=None,
        twitter_hashtags=None,
    ):
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
        self.twitter_hashtags = twitter_hashtags or ""

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in (
                "title",
                "content",
                "slug",
                "tags",
                "meta_description",
                "featured_image",
                "created_at",
                "updated_at",
                "seo_keywords",
                "affiliate_links",
                "monetization_data",
                "twitter_hashtags",
            )
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            slug=data.get("slug", ""),
            tags=data.get("tags", []),
            meta_description=data.get("meta_description", ""),
            featured_image=data.get("featured_image", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            seo_keywords=data.get("seo_keywords", []),
            affiliate_links=data.get("affiliate_links", []),
            monetization_data=data.get("monetization_data", {}),
            twitter_hashtags=data.get("twitter_hashtags", ""),
        )

    @classmethod
    def from_markdown_file(cls, md_file_path: Path, slug: str = None) -> "BlogPost":
        """Recover a post from markdown.

        FIX 1: created_at/updated_at were datetime.now(). That (a) made every
        recovered post count toward VelocityController.today_count(), (b)
        reset the corpus's apparent domain age, and (c) stamped old content
        as freshly updated — a fake-freshness signal. Use the file's mtime.
        FIX 2: meta description is now plain text (markdown/code/images
        stripped) and never taken from a code fence.
        """
        path = Path(md_file_path)
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        title, body = "Untitled Post", content
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            body = "\n".join(lines[1:]).strip()
        slug = slug or cls._create_slug_static(title)
        ts = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        return cls(
            title=title,
            content=body,
            slug=slug,
            tags=["recovered", "blog"],
            meta_description=cls._meta_from_body(body, title),
            featured_image=f"/static/images/{slug}.jpg",
            created_at=ts,
            updated_at=ts,
            seo_keywords=[],
            monetization_data={
                "ad_slots": 3,
                "affiliate_count": 0,
                "review_status": "recovered_from_markdown",
                "needs_quality_review": True,
            },
        )

    @staticmethod
    def _meta_from_body(body: str, title: str) -> str:
        text = re.sub(r"```.*?```", "", body, flags=re.S)
        for para in text.split("\n\n"):
            p = para.strip()
            if not p or p.startswith(("#", ">", "|", "!", "<", "-", "*", "1.")):
                continue
            p = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", p)
            p = re.sub(r"[*_`]", "", p)
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) < 50:
                continue
            return p if len(p) <= 160 else p[:157].rsplit(" ", 1)[0] + "..."
        return f"Blog post about {title}"

    @staticmethod
    def _create_slug_static(title: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
        return slug[:50]
