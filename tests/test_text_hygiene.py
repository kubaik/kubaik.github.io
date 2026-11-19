"""Regression tests for the "isn&#x27;t" double-escaping bug.

Run:  python -m pytest tests/test_text_hygiene.py -q
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_hygiene as th  # noqa: E402


# ---- normalisation ---------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("capacity isn&#x27;t planned", "capacity isn't planned"),
        ("capacity isn&amp;#x27;t planned", "capacity isn't planned"),  # double
        ("a &amp;amp;amp; b", "a & b"),  # triple
        ("they&#39;re &quot;fine&quot;", 'they\'re "fine"'),
        ("R&D and AT&T", "R&D and AT&T"),  # bare ampersands untouched
        ("plain text", "plain text"),
        ("zero\u200bwidth  and\n\nspaces", "zerowidth and spaces"),
    ],
)
def test_normalize_text(raw, expected):
    assert th.normalize_text(raw) == expected


def test_normalize_is_idempotent_and_safe_on_non_strings():
    once = th.normalize_text("x &amp;#x27;y&amp;#x27;")
    assert th.normalize_text(once) == once
    assert th.normalize_text(None) is None
    assert th.normalize_list(["a&#x27;b", "", "c"]) == ["a'b", "c"]


def test_clean_post_dict_leaves_content_alone():
    data = {
        "title": "Don&#x27;t",
        "meta_description": "ok",
        "tags": ["A&amp;B"],
        "content": "```html\n&lt;div&gt;\n```",
    }
    cleaned, changed = th.clean_post_dict(data)
    assert cleaned["title"] == "Don't"
    assert cleaned["tags"] == ["A&B"]
    assert cleaned["content"] == data["content"]  # code sample preserved
    assert set(changed) == {"title", "tags"}


# ---- built-HTML detection / repair ----------------------------------------
def test_detects_double_escape_but_not_correct_single_escape():
    assert th.find_double_escaped("<p>isn&amp;#x27;t</p>")
    assert not th.find_double_escaped("<p>isn&#x27;t</p>")  # correct
    assert not th.find_double_escaped("<p>AT&amp;T</p>")  # correct
    assert not th.find_double_escaped('<a href="?a=1&amp;b=2">x</a>')  # correct


def test_code_blocks_that_document_entities_are_not_flagged_or_touched():
    page = "<pre><code>&amp;lt;div&amp;gt;</code></pre><p>isn&amp;#x27;t</p>"
    assert len(th.find_double_escaped(page)) == 1
    fixed, n = th.repair_html_text(page)
    assert n == 1
    assert "<pre><code>&amp;lt;div&amp;gt;</code></pre>" in fixed
    assert "isn&#x27;t" in fixed


def test_repair_posts_json():
    raw = json.dumps([{"title": "x", "meta_description": "isn&#x27;t"}])
    fixed, n = th.repair_posts_json_text(raw)
    assert n == 1 and json.loads(fixed)[0]["meta_description"] == "isn't"


def test_assert_no_double_escaping_gate(tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "index.html").write_text("<p>isn&#x27;t</p>")
    th.assert_no_double_escaping(tmp_path)  # passes
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "index.html").write_text("<p>isn&amp;#x27;t</p>")
    with pytest.raises(RuntimeError, match="double-escaped"):
        th.assert_no_double_escaping(tmp_path)


def test_fix_sources_roundtrip(tmp_path):
    d = tmp_path / "slug-a"
    d.mkdir()
    (d / "post.json").write_text(
        json.dumps({"title": "A", "meta_description": "isn&#x27;t"})
    )
    assert len(th.scan_sources(tmp_path)) == 1
    th.fix_sources(tmp_path)
    assert json.loads((d / "post.json").read_text())["meta_description"] == "isn't"
    assert th.scan_sources(tmp_path) == []


# ---- the actual generator: end-to-end single-escape guarantee -------------
def test_plain_excerpt_returns_plain_text_and_jinja_escapes_once():
    ssg = pytest.importorskip("static_site_generator")
    from jinja2 import Environment, BaseLoader

    excerpt = ssg._plain_excerpt(
        "On-call rotations erode delivery when capacity isn't planned.", "", "t"
    )
    assert "&#" not in excerpt and "&amp;" not in excerpt  # plain text, no escaping
    html = (
        Environment(loader=BaseLoader(), autoescape=True)
        .from_string('<meta content="{{ d }}"><p>{{ d }}</p>')
        .render(d=excerpt)
    )
    assert not th.find_double_escaped(html)  # escaped exactly once


def test_plain_excerpt_cleans_prior_entities():
    ssg = pytest.importorskip("static_site_generator")
    out = ssg._plain_excerpt(
        "capacity isn&amp;#x27;t planned and it hurts a lot.", "", "t"
    )
    assert out == "capacity isn't planned and it hurts a lot."
