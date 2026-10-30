import re
import subprocess
from datetime import datetime
from pathlib import Path


class BlogPost:
    """Post model.

    FIX (root cause of truncated titles live on the site, e.g.
    "Staging won't catch prod bugs. Make…"): blog_system.save_post() writes
    `full_title` into post.json, but this class neither stored it in
    __init__ nor round-tripped it in to_dict()/from_dict(). Every build
    therefore reloaded posts WITHOUT their full title and rendered the
    display-truncated one in <title>, og:title, H1 and JSON-LD headline.
    """

    _FIELDS = (
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
        "full_title",
    )

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
        full_title=None,
    ):
        self.title = title
        self.full_title = full_title or title
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

    @property
    def display_title(self) -> str:
        """Title for <title>/og/H1/schema: the untruncated one if known."""
        return self.full_title or self.title

    def to_dict(self):
        return {k: getattr(self, k) for k in self._FIELDS}

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
            full_title=data.get("full_title") or data.get("title", ""),
        )

    @staticmethod
    def _first_commit_iso(path: Path):
        """First-commit time of a file. File mtime is useless in CI: every
        GitHub Actions checkout stamps all files with the checkout time, so
        mtime-based recovery made old posts look published today (inflating
        VelocityController.today_count() and faking freshness)."""
        try:
            out = (
                subprocess.run(
                    [
                        "git",
                        "log",
                        "--diff-filter=A",
                        "--follow",
                        "--format=%aI",
                        "--",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                .stdout.strip()
                .splitlines()
            )
            return out[-1] if out else None
        except Exception:
            return None

    @classmethod
    def from_markdown_file(cls, md_file_path: Path, slug: str = None) -> "BlogPost":
        path = Path(md_file_path)
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        title, body = "Untitled Post", content
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            body = "\n".join(lines[1:]).strip()
        slug = slug or cls._create_slug_static(title)
        ts = (
            cls._first_commit_iso(path)
            or datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        )

        return cls(
            title=title,
            full_title=title,
            content=body,
            slug=slug,
            tags=[],  # was ["recovered","blog"]: junk tags became tag pages
            meta_description=cls._meta_from_body(body, title),
            featured_image=f"/static/images/{slug}.jpg",
            created_at=ts,
            updated_at=ts,
            seo_keywords=[],
            monetization_data={
                "ad_slots": 0,  # recovered posts must not carry ads until reviewed
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
        if len(slug) <= 60:
            return slug
        return slug[:60].rsplit("-", 1)[0]  # cut on a word boundary
