"""Red-first tests for markdown chunking.

chunk_markdown is the preprocessor that turns a single Markdown
file into retrieval-ready chunks. Per-H2 sectioning, H1+heading
denormalization, small-file single-chunk, sliding-window on
oversized sections.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from familiar_connect.context.providers.content_search.retrieval import (
    HARD_CHUNK_TOKEN_CAP,
    SMALL_FILE_WORD_THRESHOLD,
    chunk_markdown,
)


class TestSingleChunkPaths:
    def test_empty_input_no_chunks(self) -> None:
        assert chunk_markdown("", rel_path="x.md", mtime=0.0) == []

    def test_whitespace_only_no_chunks(self) -> None:
        assert chunk_markdown("   \n\n  ", rel_path="x.md", mtime=0.0) == []

    def test_small_file_single_chunk(self) -> None:
        """Tiny people file stays whole.

        Denormalization with heading wouldn't help because the file
        is already short enough to scan.
        """
        text = "# Alice\n\nAlice likes ska."
        chunks = chunk_markdown(text, rel_path="people/alice.md", mtime=1.0)
        assert len(chunks) == 1
        assert chunks[0].rel_path == "people/alice.md"
        assert chunks[0].mtime == pytest.approx(1.0)
        assert "Alice likes ska" in chunks[0].text
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_file_with_no_headings_single_chunk(self) -> None:
        text = "Just some prose with no headings at all." * 3
        chunks = chunk_markdown(text, rel_path="notes.md", mtime=0.0)
        assert len(chunks) == 1
        assert not chunks[0].heading_path


class TestH2Sectioning:
    def test_two_h2_sections_two_chunks(self) -> None:
        # build a file that exceeds the small-file threshold so
        # sectioning fires; repeat filler per section.
        filler = "alpha beta gamma delta epsilon zeta eta theta iota " * 40
        text = (
            "# Alice\n\n"
            f"{filler}\n\n"
            "## Likes\n\n"
            f"She enjoys ska and old citadels. {filler}\n\n"
            "## Dislikes\n\n"
            f"Can't stand modern pop. {filler}"
        )
        chunks = chunk_markdown(text, rel_path="people/alice.md", mtime=0.0)
        assert len(chunks) >= 2
        likes = next(c for c in chunks if "ska and old citadels" in c.text)
        dislikes = next(c for c in chunks if "modern pop" in c.text)
        # H1 prepended to heading_path so embedding captures file context
        assert "Alice" in likes.heading_path
        assert "Likes" in likes.heading_path
        assert "Alice" in dislikes.heading_path
        assert "Dislikes" in dislikes.heading_path

    def test_heading_path_appears_in_chunk_text(self) -> None:
        """Denormalised embedding input.

        Heading path prepended to body so short sections still match
        on their topic even when the body alone doesn't mention it.
        """
        filler = "word " * 200
        text = (
            f"# Citadels\n\n{filler}\n\n"
            f"## The Old Citadel\n\nRuined and haunted. {filler}"
        )
        chunks = chunk_markdown(text, rel_path="lore/citadels.md", mtime=0.0)
        haunted = next(c for c in chunks if "Ruined and haunted" in c.text)
        # heading path content appears in the embedding input
        assert "Citadels" in haunted.text
        assert "Old Citadel" in haunted.text

    def test_preamble_before_first_h2_is_its_own_chunk(self) -> None:
        filler = "word " * 200
        text = (
            f"# Alice\n\nAn overview paragraph. {filler}\n\n## Likes\n\nSka. {filler}"
        )
        chunks = chunk_markdown(text, rel_path="people/alice.md", mtime=0.0)
        # preamble ("An overview paragraph") becomes its own chunk tagged
        # with just the H1
        preamble = next(c for c in chunks if "overview paragraph" in c.text)
        assert preamble.heading_path == "Alice"


class TestSlidingWindow:
    def test_oversize_section_produces_multiple_chunks(self) -> None:
        """A single H2 exceeding the hard cap is windowed."""
        # ~5x cap of 'a' chars -> definitely needs multiple windows
        huge_body = "word " * (HARD_CHUNK_TOKEN_CAP * 5)
        text = f"# Topic\n\n## Giant\n\n{huge_body}"
        chunks = chunk_markdown(text, rel_path="topics/giant.md", mtime=0.0)
        assert len(chunks) >= 2
        # all fragments share the same heading path
        heading_paths = {c.heading_path for c in chunks}
        assert heading_paths == {"Topic > Giant"}
        # chunks respect the hard cap (allow a little slack)
        for c in chunks:
            assert c.token_count <= HARD_CHUNK_TOKEN_CAP + 50

    def test_sliding_windows_overlap(self) -> None:
        """Adjacent sliding windows overlap.

        window=800, stride=600 → 200-token overlap so an important
        sentence near a boundary isn't orphaned.
        """
        huge_body = " ".join(f"sentence{i}." for i in range(4000))
        text = f"# Topic\n\n## Giant\n\n{huge_body}"
        chunks = chunk_markdown(text, rel_path="topics/giant.md", mtime=0.0)
        assert len(chunks) >= 2
        # char_end of chunk N > char_start of chunk N+1 (overlap)
        for a, b in pairwise(chunks):
            if a.heading_path == b.heading_path:
                assert a.char_end > b.char_start


class TestSmallFileThreshold:
    def test_threshold_constant_reasonable(self) -> None:
        # documented threshold, tests pin it so a future change is visible
        assert SMALL_FILE_WORD_THRESHOLD == 300
