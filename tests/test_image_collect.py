"""Tests for collect_images in bot.py."""

from __future__ import annotations

from types import SimpleNamespace

from familiar_connect.bot import collect_images


def _att(url: str, filename: str, content_type: str = "image/png") -> object:
    return SimpleNamespace(url=url, filename=filename, content_type=content_type)


def _embed(image_url: str) -> object:
    return SimpleNamespace(image=SimpleNamespace(url=image_url))


def _embed_no_image() -> object:
    return SimpleNamespace(image=None)


# ---------------------------------------------------------------------------
# attachments
# ---------------------------------------------------------------------------


def test_collects_attachment_and_injects_placeholder() -> None:
    att = _att("http://cdn.example.com/cat.png", "cat.png")
    content, images = collect_images(content="hello", attachments=[att], embeds=[])
    assert images == {"img_0": "http://cdn.example.com/cat.png"}
    assert "[image: img_0 (cat.png)]" in content


def test_non_image_attachment_excluded() -> None:
    att = SimpleNamespace(
        url="http://x.com/doc.pdf", filename="doc.pdf", content_type="application/pdf"
    )
    content, images = collect_images(content="hi", attachments=[att], embeds=[])
    assert images == {}
    assert content == "hi"


# ---------------------------------------------------------------------------
# embeds
# ---------------------------------------------------------------------------


def test_collects_embed_image() -> None:
    embed = _embed("http://cdn.example.com/preview.jpg")
    content, images = collect_images(content="", attachments=[], embeds=[embed])
    assert len(images) == 1
    assert "preview.jpg" in next(iter(images.values()))
    assert "[image: img_0" in content


def test_embed_without_image_ignored() -> None:
    embed = _embed_no_image()
    content, images = collect_images(content="hello", attachments=[], embeds=[embed])
    assert images == {}
    assert content == "hello"


# ---------------------------------------------------------------------------
# inline URLs
# ---------------------------------------------------------------------------


def test_collects_inline_url() -> None:
    url = "https://i.imgur.com/abc.jpg"
    content, images = collect_images(
        content=f"check this {url}", attachments=[], embeds=[]
    )
    assert len(images) == 1
    assert url in images.values()
    assert "[image: img_0 (abc.jpg)]" in content


def test_no_images_returns_content_unchanged_empty_dict() -> None:
    content, images = collect_images(content="just text", attachments=[], embeds=[])
    assert content == "just text"
    assert images == {}


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------


def test_dedupes_url_present_as_attachment_and_inline() -> None:
    url = "http://cdn.example.com/cat.png"
    att = _att(url, "cat.png")
    _, images = collect_images(content=f"look: {url}", attachments=[att], embeds=[])
    # same URL only gets one entry
    assert len(images) == 1
    assert url in images.values()


def test_multiple_images_get_sequential_ids() -> None:
    att0 = _att("http://cdn.example.com/a.png", "a.png")
    att1 = _att("http://cdn.example.com/b.jpeg", "b.jpeg")
    content, images = collect_images(
        content="hello", attachments=[att0, att1], embeds=[]
    )
    assert set(images.keys()) == {"img_0", "img_1"}
    assert "[image: img_0 (a.png)]" in content
    assert "[image: img_1 (b.jpeg)]" in content
