import json
import html as _html_module
from datetime import datetime
from typing import List


def _esc(text: str) -> str:
    """HTML-escape a string for safe use in attribute values."""
    return _html_module.escape(str(text or ""), quote=True)


def _normalize_iso_date(dt_str: str) -> str:
    """Strip microseconds; ensure UTC offset. Google rejects microsecond precision."""
    if not dt_str:
        return dt_str
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except Exception:
        if "." in dt_str:
            base, frac = dt_str.split(".", 1)
            tz = ""
            for sep in ("+", "-"):
                if sep in frac:
                    tz = sep + frac.split(sep, 1)[1]
                    break
            return base + (tz or "+00:00")
        return dt_str


class SEOOptimizer:
    """SEO, structured data, and AdSense integration."""

    def __init__(self, config):
        self.config = config
        self.google_analytics_id = config.get("google_analytics_id", "")
        self.google_adsense_id = config.get("google_adsense_id", "")
        # google_search_console_key must be the HTML-meta verification token,
        # NOT an API key. The config currently stores an API key here — leave
        # it blank until the correct token is obtained from Search Console.
        raw_gsc = config.get("google_search_console_key", "")
        # API keys start with "AIza" — skip them silently so we never emit a
        # broken verification tag that Search Console would reject.
        self.google_search_console_verification = (
            raw_gsc if raw_gsc and not raw_gsc.startswith("AIza") else ""
        )

    # ------------------------------------------------------------------ #
    #  GLOBAL HEAD TAGS                                                    #
    # ------------------------------------------------------------------ #

    def generate_global_meta_tags(self) -> str:
        """
        Emit: AdSense account tag, GSC verification, GA4 snippet, AdSense loader.

        IMPORTANT LOAD ORDER (required by Consent Mode v2 and AdSense policy):
          1. consent.js  ← already placed as first <script> in every template
          2. google-adsense-account meta
          3. GA4 (gtag.js) — respects consent state set by consent.js
          4. AdSense loader — respects consent state set by consent.js

        consent.js calls gtag('consent','default',...) BEFORE GA4/AdSense
        scripts execute because those are `async` and consent.js is sync.
        Do NOT move GA4 or AdSense above consent.js.
        """
        parts: list[str] = []

        adsense_id = self._fmt_adsense_id()

        if adsense_id:
            parts.append(
                f'    <meta name="google-adsense-account" content="{adsense_id}">'
            )

        if self.google_search_console_verification:
            parts.append(
                f'    <meta name="google-site-verification" '
                f'content="{_esc(self.google_search_console_verification)}">'
            )

        # GA4 — async so consent.js default state is already in dataLayer
        if self.google_analytics_id:
            ga_id = _esc(self.google_analytics_id)
            parts.append(
                f"    <!-- Google Analytics 4 -->\n"
                f'    <script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>\n'
                f"    <script>\n"
                f"    window.dataLayer = window.dataLayer || [];\n"
                f"    function gtag(){{dataLayer.push(arguments);}}\n"
                f"    gtag('js', new Date());\n"
                f"    gtag('config', '{ga_id}', {{anonymize_ip: true}});\n"
                f"    </script>"
            )

        # AdSense loader — async; respects consent default pushed by consent.js
        if adsense_id:
            parts.append(
                f"    <!-- Google AdSense -->\n"
                f'    <script async '
                f'src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={adsense_id}" '
                f'crossorigin="anonymous"></script>'
            )

        return "\n".join(parts) + "\n" if parts else ""

    # ------------------------------------------------------------------ #
    #  PER-POST META TAGS                                                  #
    # ------------------------------------------------------------------ #

    def generate_meta_tags(self, post) -> str:
        """
        Return ONLY tags that the post template does NOT already emit.

        The POST_TMPL in static_site_generator already emits:
          - <meta name="description">
          - og:title, og:description, og:url, og:site_name, og:locale,
            og:type, og:image, og:image:alt, og:image:width/height
          - twitter:card, twitter:title, twitter:description, twitter:image,
            twitter:site
          - <link rel="canonical">
          - article:author, article:published_time, article:modified_time,
            article:section
          - <meta name="author">

        This method adds ONLY the tags that template does NOT emit to avoid
        duplicates that confuse crawlers and break Google's structured data
        validator.
        """
        parts: list[str] = []

        # robots directive (not in template)
        parts.append(
            '    <meta name="robots" content="index, follow, '
            'max-image-preview:large, max-snippet:-1, max-video-preview:-1">'
        )

        # keywords (template emits this tag, but only when seo_keywords is
        # truthy on the dict — generate_meta_tags is called independently and
        # its output is appended *after* the template block, so we skip here
        # to avoid the double-emit confirmed in the audit).
        # The template already handles keywords via:
        #   {% if post.seo_keywords %}<meta name="keywords" ...>{% endif %}

        return "\n".join(parts) + "\n" if parts else ""

    # ------------------------------------------------------------------ #
    #  HOMEPAGE META TAGS                                                  #
    # ------------------------------------------------------------------ #

    def generate_homepage_meta_tags(self) -> str:
        """
        Tags for the homepage that the INDEX_TMPL does NOT already emit.
        The template already emits og:*, twitter:*, canonical, and description.
        We add robots directive only.
        """
        return (
            '    <meta name="robots" content="index, follow, '
            'max-image-preview:large, max-snippet:-1, max-video-preview:-1">\n'
        )

    # ------------------------------------------------------------------ #
    #  AD UNITS                                                            #
    # ------------------------------------------------------------------ #

    def generate_adsense_ad(self, slot_type: str = "display", slot_id: str = None) -> str:
        """
        Return a bare <ins> + push() block with NO wrapping <div>.

        The template already wraps each ad in:
            <div class="ad-header">…</div>
            <div class="ad-inline">…</div>
            <div class="ad-middle">…</div>
            <div class="ad-footer">…</div>

        Previously this method also emitted a <div class="ad-{slot_type}">
        wrapper, causing the rendered HTML to become:
            <div class="ad-header">          ← from template
              <div class="ad-header" …>      ← from this method  ← BUG
                <ins …></ins>
              </div>
            </div>

        That double-nesting breaks the CSS :not(:empty) rules and causes CLS
        because the outer div is the one the CSS measures but the inner one
        is the one AdSense fills.  Removed the wrapper entirely.

        For pre-approval (no slot IDs assigned yet) we use Auto Ads format
        (data-ad-format="auto" + data-full-width-responsive="true").
        Once real slot IDs exist, pass them via config and they will be
        used instead.
        """
        adsense_id = self._fmt_adsense_id()
        if not adsense_id:
            return f"<!-- AdSense slot '{slot_type}' — no publisher ID configured -->"

        slot_map = self.config.get("adsense_slots", {})
        resolved_slot_id = slot_id or slot_map.get(slot_type, "")
        data_slot_attr = f'\n         data-ad-slot="{_esc(resolved_slot_id)}"' if resolved_slot_id else ""

        # Use Auto Ads (responsive) when no slot ID — safest for approval phase
        if not resolved_slot_id:
            return (
                f'<ins class="adsbygoogle"\n'
                f'     style="display:block"\n'
                f'     data-ad-client="{adsense_id}"\n'
                f'     data-ad-format="auto"\n'
                f'     data-full-width-responsive="true"></ins>\n'
                f"<script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>"
            )

        # Fixed-slot format once slot IDs are known
        style_map = {
            "header": "display:inline-block;width:728px;height:90px",
            "middle": "display:block",
            "footer": "display:block",
            "inline": "display:block",
        }
        style = style_map.get(slot_type, "display:block")
        return (
            f'<ins class="adsbygoogle"\n'
            f'     style="{style}"\n'
            f'     data-ad-client="{adsense_id}"{data_slot_attr}\n'
            f'     data-ad-format="auto"\n'
            f'     data-full-width-responsive="true"></ins>\n'
            f"<script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>"
        )

    # ------------------------------------------------------------------ #
    #  STRUCTURED DATA                                                     #
    # ------------------------------------------------------------------ #

    def generate_structured_data(self, post) -> str:
        """
        Emit a minimal BlogPosting schema.

        NOTE: static_site_generator._generate_article_schema() already emits
        a richer Article + BreadcrumbList @graph block for every post page.
        This method is called alongside it.  To avoid duplicate @type:Article
        nodes that confuse Google's Rich Results Test, we emit BlogPosting
        (a subtype) so the two schemas are distinguishable.  Ideally you would
        consolidate to a single schema block — tracked as a future improvement.
        """
        base_url = self.config.get("base_url", "")
        base_path = self.config.get("base_path", "")
        post_url = f"{base_url}/{post.slug}/"

        schema = {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "@id": f"{post_url}#blogposting",
            "headline": post.title,
            "description": post.meta_description or "",
            "author": {
                "@type": "Person",
                "@id": f"{base_url}/about/#author",
                "name": "Kubai Kevin",
                "url": f"{base_url}/about/",
                "sameAs": [
                    "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
                    "https://twitter.com/KubaiKevin",
                ],
            },
            "publisher": {
                "@type": "Organization",
                "name": self.config.get("site_name", "Tech Blog"),
                "url": f"{base_url}/",
            },
            "datePublished": _normalize_iso_date(post.created_at),
            "dateModified": _normalize_iso_date(post.updated_at),
            "url": post_url,
            "mainEntityOfPage": {"@type": "WebPage", "@id": post_url},
        }

        if hasattr(post, "seo_keywords") and post.seo_keywords:
            schema["keywords"] = (
                ", ".join(post.seo_keywords)
                if isinstance(post.seo_keywords, list)
                else post.seo_keywords
            )

        return (
            f'<script type="application/ld+json">\n'
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n"
            f"</script>"
        )

    def generate_organization_schema(self) -> str:
        base_url = self.config.get("base_url", "")
        social = self.config.get("social_accounts", {})
        same_as = []
        if social.get("twitter"):
            same_as.append(
                f"https://twitter.com/{social['twitter'].lstrip('@')}")
        if social.get("linkedin") and not social["linkedin"].startswith("your-"):
            same_as.append(f"https://www.linkedin.com/in/{social['linkedin']}")

        schema = {
            "@context": "https://schema.org",
            "@type": "Organization",
            "@id": f"{base_url}/#organization",
            "name": self.config.get("site_name", "Tech Blog"),
            "description": self.config.get("site_description", ""),
            "url": f"{base_url}/",
        }
        if same_as:
            schema["sameAs"] = same_as

        return (
            f'<script type="application/ld+json">\n'
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n"
            f"</script>"
        )

    def generate_website_schema(self) -> str:
        base_url = self.config.get("base_url", "")
        schema = {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "@id": f"{base_url}/#website",
            "name": self.config.get("site_name", "Tech Blog"),
            "description": self.config.get("site_description", ""),
            "url": f"{base_url}/",
            "publisher": {"@id": f"{base_url}/#organization"},
        }
        return (
            f'<script type="application/ld+json">\n'
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n"
            f"</script>"
        )

    # ------------------------------------------------------------------ #
    #  SITEMAP / ROBOTS (unused — static_site_generator owns these)       #
    # ------------------------------------------------------------------ #

    def generate_sitemap(self, posts) -> str:
        """Kept for backwards compat. static_site_generator._generate_sitemap
        is the canonical implementation and should be used instead."""
        base_url = self.config.get("base_url", "")
        urls = [
            f"  <url><loc>{base_url}/</loc>"
            f"<lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>"
            f"<changefreq>daily</changefreq><priority>1.0</priority></url>"
        ]
        for post in posts:
            lastmod = post.updated_at.split(
                "T")[0] if "T" in post.updated_at else post.updated_at
            urls.append(
                f"  <url><loc>{base_url}/{post.slug}/</loc>"
                f"<lastmod>{lastmod}</lastmod>"
                f"<changefreq>weekly</changefreq><priority>0.9</priority></url>"
            )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + "\n".join(urls)
            + "\n</urlset>"
        )

    def generate_robots_txt(self) -> str:
        base_url = self.config.get("base_url", "")
        return (
            "User-agent: *\nAllow: /\nDisallow: /static/admin/\n\n"
            "User-agent: Mediapartners-Google\nAllow: /\n\n"
            "User-agent: Googlebot\nAllow: /\nCrawl-delay: 1\n\n"
            f"Sitemap: {base_url}/sitemap.xml\n"
            f"Sitemap: {base_url}/rss.xml\n"
        )

    # ------------------------------------------------------------------ #
    #  SOCIAL LINKS (utility)                                              #
    # ------------------------------------------------------------------ #

    def get_social_media_links(self) -> dict:
        social = self.config.get("social_accounts", {})
        links = {}
        if social.get("twitter"):
            links["twitter"] = f"https://twitter.com/{social['twitter'].lstrip('@')}"
        if social.get("linkedin") and not social["linkedin"].startswith("your-"):
            links["linkedin"] = f"https://www.linkedin.com/in/{social['linkedin']}"
        return links

    # ------------------------------------------------------------------ #
    #  INTERNAL                                                            #
    # ------------------------------------------------------------------ #

    def _fmt_adsense_id(self) -> str:
        aid = (self.google_adsense_id or "").strip()
        if not aid:
            return ""
        if not aid.startswith("ca-pub-"):
            aid = f"ca-pub-{aid}"
        return aid
