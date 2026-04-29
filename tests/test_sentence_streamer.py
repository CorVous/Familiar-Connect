"""Tests for :mod:`familiar_connect.sentence_streamer`.

Sentence-boundary aggregator between LLM stream and TTS. Splits on
``.``/``!``/``?`` followed by whitespace; respects common
abbreviations and single-letter initials. ``flush`` drains whatever
remains when the LLM stream ends.
"""

from __future__ import annotations

import pytest

from familiar_connect.sentence_streamer import SentenceStreamer


class TestSimpleSplitting:
    def test_no_boundary_yet_buffers(self) -> None:
        s = SentenceStreamer()
        assert s.feed("Hello") == []
        assert s.feed(", world") == []

    def test_period_then_space_emits_sentence(self) -> None:
        s = SentenceStreamer()
        out = s.feed("Hello, world. ")
        assert out == ["Hello, world."]

    def test_period_at_end_buffers_until_space_or_flush(self) -> None:
        """Trailing ``.`` alone can't split — could be an abbrev mid-stream."""
        s = SentenceStreamer()
        assert s.feed("Hello, world.") == []
        # Flush drains the tail.
        assert s.flush() == "Hello, world."

    def test_question_and_exclamation_are_boundaries(self) -> None:
        s = SentenceStreamer()
        assert s.feed("Really? ") == ["Really?"]
        assert s.feed("Wow! ") == ["Wow!"]

    def test_two_sentences_in_one_delta(self) -> None:
        s = SentenceStreamer()
        out = s.feed("First. Second. ")
        assert out == ["First.", "Second."]

    def test_split_across_multiple_deltas(self) -> None:
        s = SentenceStreamer()
        assert s.feed("Hello") == []
        assert s.feed(", ") == []
        assert s.feed("world") == []
        assert s.feed(".") == []
        assert s.feed(" How") == ["Hello, world."]
        assert s.feed(" are you?") == []
        assert s.feed(" ") == ["How are you?"]


class TestAbbreviations:
    """Abbreviation-aware: don't split inside ``Mr.``, ``Dr.``, etc."""

    @pytest.mark.parametrize(
        "abbrev",
        ["Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Sr.", "Jr.", "Prof.", "Rev."],
    )
    def test_title_abbreviations_do_not_split(self, abbrev: str) -> None:
        s = SentenceStreamer()
        assert s.feed(f"{abbrev} Smith arrived. ") == [f"{abbrev} Smith arrived."]

    def test_etc_does_not_split(self) -> None:
        s = SentenceStreamer()
        assert s.feed("Apples, pears, etc. are fine. ") == [
            "Apples, pears, etc. are fine."
        ]

    def test_eg_and_ie_do_not_split(self) -> None:
        s = SentenceStreamer()
        assert s.feed("Some fruits, e.g. apples, work. ") == [
            "Some fruits, e.g. apples, work."
        ]
        s2 = SentenceStreamer()
        assert s2.feed("That is, i.e. always. ") == ["That is, i.e. always."]

    def test_single_letter_initial_does_not_split(self) -> None:
        """``J. K. Rowling`` is one sentence, not three."""
        s = SentenceStreamer()
        assert s.feed("J. K. Rowling wrote it. ") == ["J. K. Rowling wrote it."]


class TestFlush:
    def test_flush_drains_partial_buffer_verbatim(self) -> None:
        """Trailing partial text — including spaces — survives flush.

        Preserves prosody for replies without terminal punctuation.
        """
        s = SentenceStreamer()
        assert s.feed("d0 d1 d2 ") == []
        assert s.flush() == "d0 d1 d2 "

    def test_flush_after_emitted_sentence_returns_remainder(self) -> None:
        s = SentenceStreamer()
        assert s.feed("First. ") == ["First."]
        assert s.feed("partial") == []
        assert s.flush() == "partial"

    def test_flush_resets_buffer(self) -> None:
        s = SentenceStreamer()
        s.feed("hi")
        assert s.flush() == "hi"
        assert not s.flush()

    def test_flush_with_trailing_period_no_space(self) -> None:
        """Period without trailing space is held; flush drains as-is."""
        s = SentenceStreamer()
        assert s.feed("Done.") == []
        assert s.flush() == "Done."


class TestEdgeCases:
    def test_empty_delta_is_noop(self) -> None:
        s = SentenceStreamer()
        assert s.feed("") == []
        assert not s.flush()

    def test_silent_sentinel_never_emits(self) -> None:
        """``<silent>`` has no terminal punctuation: held for caller to gate."""
        s = SentenceStreamer()
        assert s.feed("<silent>") == []
        assert s.flush() == "<silent>"

    def test_consecutive_punctuation_collapses_into_one_boundary(self) -> None:
        """``Wait?!`` is a single sentence ending."""
        s = SentenceStreamer()
        assert s.feed("Wait?! ") == ["Wait?!"]

    def test_newline_acts_as_terminator_whitespace(self) -> None:
        s = SentenceStreamer()
        assert s.feed("Done.\nNext") == ["Done."]
