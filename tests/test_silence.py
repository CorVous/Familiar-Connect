"""Tests for :mod:`familiar_connect.silence` SilentDetector."""

from __future__ import annotations

import pytest

from familiar_connect.silence import SILENT_TOKEN, SilentDetector


class TestSilentDetector:
    def test_pending_until_first_chars(self) -> None:
        d = SilentDetector()
        assert d.feed("") is None
        assert d.decided is None

    def test_detects_full_token_in_one_delta(self) -> None:
        d = SilentDetector()
        assert d.feed(SILENT_TOKEN) is True
        assert d.decided is True

    def test_detects_token_split_across_deltas(self) -> None:
        d = SilentDetector()
        assert d.feed("<sil") is None
        assert d.feed("ent>") is True

    def test_tolerates_leading_whitespace(self) -> None:
        d = SilentDetector()
        assert d.feed("   ") is None
        assert d.feed("<silent>") is True

    def test_rejects_token_not_at_prefix(self) -> None:
        """Mid-reply ``<silent>`` is content, not a gate."""
        d = SilentDetector()
        assert d.feed("Sure, ") is False
        assert d.feed("<silent>") is False

    def test_rejects_normal_content(self) -> None:
        d = SilentDetector()
        # 'H' diverges immediately from '<'
        assert d.feed("Hello world") is False

    def test_rejects_when_diverges_mid_token(self) -> None:
        d = SilentDetector()
        assert d.feed("<sil") is None
        # 'k' diverges from expected 'e'
        assert d.feed("k") is False

    def test_decision_latches(self) -> None:
        d = SilentDetector()
        assert d.feed("<silent>") is True
        # subsequent feeds return cached decision without re-inspecting
        assert d.feed("anything goes here") is True

    def test_decision_latches_for_speak(self) -> None:
        d = SilentDetector()
        assert d.feed("Hi ") is False
        assert d.feed("<silent>") is False

    def test_long_run_of_whitespace_stays_pending(self) -> None:
        """Pure whitespace can't decide; existing empty-reply guard handles."""
        d = SilentDetector()
        assert d.feed("\n\n\n   ") is None
        assert d.decided is None

    @pytest.mark.parametrize(
        ("deltas", "expected"),
        [
            (["<", "s", "i", "l", "e", "n", "t", ">"], True),
            (["  <silent>"], True),
            (["<silently I disagree>"], False),
            (["  Hello"], False),
        ],
    )
    def test_parameterized(
        self,
        deltas: list[str],
        expected: bool,  # noqa: FBT001 — pytest parametrize passes positionally
    ) -> None:
        d = SilentDetector()
        result: bool | None = None
        for delta in deltas:
            result = d.feed(delta)
        assert result is expected
