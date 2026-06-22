"""Tests for crash-safe ``{placeholder}`` fill (``prompt_fill``)."""

from __future__ import annotations

from familiar_connect.prompt_fill import fill_placeholders


def test_fills_known_placeholder() -> None:
    assert fill_placeholders("hi {name}", name="Sapphire") == "hi Sapphire"


def test_unknown_placeholder_passes_through() -> None:
    assert fill_placeholders("hi {name} {other}", name="Sapphire") == "hi Sapphire {other}"


def test_missing_placeholder_passes_through() -> None:
    # template lacks the supplied key — no error, unchanged
    assert fill_placeholders("plain text", name="Sapphire") == "plain text"


def test_stray_braces_pass_through() -> None:
    assert fill_placeholders("a { b } c {name}", name="X") == "a { b } c X"


def test_injected_value_is_not_re_expanded() -> None:
    """Single pass: a value containing another key's token stays literal.

    Order-independence guard — chained per-key replacement would re-scan
    ``a``'s injected ``{b}`` and expand it to ``X`` ("X X"). One pass
    fills each token exactly once.
    """
    assert fill_placeholders("{a} {b}", a="{b}", b="X") == "{b} X"
