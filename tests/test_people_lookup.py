"""Red-first tests for deterministic people-lookup tier.

Tier 1 of the three-tier ContentSearchProvider. No LLM involvement;
speaker's ``people/<slug>.md`` and any file matching a name mentioned
in the utterance are always included verbatim. The correctness floor
that guarantees the familiar never forgets someone it has notes on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.providers.content_search.people_lookup import (
    PEOPLE_LOOKUP_PRIORITY,
    PEOPLE_LOOKUP_SOURCE,
    lookup,
)
from familiar_connect.context.types import ContextRequest, Layer, Modality
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path


def _request(
    *,
    utterance: str = "hi",
    speaker: str | None = "Alice",
) -> ContextRequest:
    return ContextRequest(
        familiar_id="aria",
        channel_id=100,
        guild_id=1,
        speaker=speaker,
        utterance=utterance,
        modality=Modality.text,
        budget_tokens=2048,
        deadline_s=10.0,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


# ---------------------------------------------------------------------------
# Speaker file — always loaded when present
# ---------------------------------------------------------------------------


class TestSpeakerFile:
    def test_speaker_file_included_when_present(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "Alice likes ska.")

        contributions = lookup(store, _request())

        assert len(contributions) == 1
        c = contributions[0]
        assert c.layer is Layer.content
        assert c.priority == PEOPLE_LOOKUP_PRIORITY
        assert c.source == PEOPLE_LOOKUP_SOURCE
        assert "Alice likes ska" in c.text

    def test_speaker_file_included_regardless_of_utterance(
        self, store: MemoryStore
    ) -> None:
        """Speaker file wins even when utterance doesn't mention them."""
        store.write_file("people/alice.md", "A note about Alice.")

        contributions = lookup(
            store, _request(utterance="what's the weather?", speaker="Alice")
        )

        assert len(contributions) == 1
        assert "A note about Alice" in contributions[0].text

    def test_no_speaker_file_no_contribution(self, store: MemoryStore) -> None:
        contributions = lookup(store, _request(utterance="hello"))
        assert contributions == []

    def test_empty_speaker_slug_ignored(self, store: MemoryStore) -> None:
        """A speaker that slugifies to empty (e.g. only punctuation) doesn't crash."""
        store.write_file("people/alice.md", "x")
        contributions = lookup(store, _request(utterance="hello", speaker="!!!"))
        assert contributions == []

    def test_speaker_slug_matches_writer_convention(self, store: MemoryStore) -> None:
        """Mixed-case + punctuation slugifies the same as memory/writer.py."""
        store.write_file("people/mary-jane.md", "MJ notes.")
        contributions = lookup(store, _request(utterance="hi", speaker="Mary Jane"))
        assert len(contributions) == 1
        assert "MJ notes" in contributions[0].text


# ---------------------------------------------------------------------------
# Mentioned names — extracted from utterance
# ---------------------------------------------------------------------------


class TestMentionedNames:
    def test_capitalized_mention_loads_file(self, store: MemoryStore) -> None:
        store.write_file("people/bob.md", "Bob the Builder.")
        contributions = lookup(
            store, _request(utterance="tell me about Bob", speaker=None)
        )
        assert len(contributions) == 1
        assert "Bob the Builder" in contributions[0].text

    def test_lowercase_mention_loads_file(self, store: MemoryStore) -> None:
        """Pass (b) reverse-match: utterance token matches filename stem."""
        store.write_file("people/bob.md", "Bob notes.")
        contributions = lookup(store, _request(utterance="where is bob?", speaker=None))
        assert len(contributions) == 1
        assert "Bob notes" in contributions[0].text

    def test_hyphenated_filename_matches_space_separated_phrase(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/bob-the-builder.md", "a handyman")
        contributions = lookup(
            store,
            _request(utterance="where is bob the builder?", speaker=None),
        )
        assert len(contributions) == 1
        assert "handyman" in contributions[0].text

    def test_no_match_no_contribution(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "A")
        store.write_file("people/bob.md", "B")
        contributions = lookup(
            store, _request(utterance="what's for lunch?", speaker=None)
        )
        assert contributions == []

    def test_speaker_first_then_mentioned(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "A")
        store.write_file("people/bob.md", "B")
        contributions = lookup(store, _request(utterance="hi Bob", speaker="Alice"))
        assert len(contributions) == 2
        assert contributions[0].text.strip() == "A"  # speaker first
        assert contributions[1].text.strip() == "B"

    def test_mention_dedupes_with_speaker(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "A")
        # speaker mentions themselves in utterance — still only one contribution
        contributions = lookup(
            store, _request(utterance="hi, I'm Alice", speaker="Alice")
        )
        assert len(contributions) == 1


# ---------------------------------------------------------------------------
# Budget handling
# ---------------------------------------------------------------------------


class TestBudget:
    def test_oversize_file_truncated(self, store: MemoryStore) -> None:
        long_text = "word " * 2000  # ~2500 tokens — well over per-file cap
        store.write_file("people/alice.md", long_text)
        contributions = lookup(store, _request())
        assert len(contributions) == 1
        # default per-file cap is 800 tokens; allow a little slack
        assert contributions[0].estimated_tokens <= 900

    def test_content_cap_drops_later_files(self, store: MemoryStore) -> None:
        """Budget overflow drops utterance-order tail, keeps speaker."""
        long_text = "word " * 1000  # truncates to per-file cap (~800 tokens)
        store.write_file("people/alice.md", long_text)
        store.write_file("people/bob.md", long_text)

        contributions = lookup(
            store,
            _request(utterance="hi Bob", speaker="Alice"),
            content_cap_tokens=900,  # leaves room for exactly one file
        )

        # speaker (alice) wins; bob dropped
        assert len(contributions) == 1
