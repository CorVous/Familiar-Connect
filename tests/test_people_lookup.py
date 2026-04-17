"""Red-first tests for deterministic people-lookup tier.

Tier 1 of the three-tier ContentSearchProvider. No LLM involvement;
speaker's ``people/<author.slug>.md`` and any file matching a name
mentioned in the utterance (resolved via ``people/_aliases.json``) are
always included verbatim. The correctness floor that guarantees the
familiar never forgets someone it has notes on.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.providers.content_search.people_lookup import (
    PEOPLE_LOOKUP_PRIORITY,
    PEOPLE_LOOKUP_SOURCE,
    lookup,
)
from familiar_connect.context.types import ContextRequest, Layer, Modality
from familiar_connect.identity import Author
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path


_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")
_BOB = Author(platform="discord", user_id="2", username="bob", display_name="Bob")


def _request(
    *,
    utterance: str = "hi",
    author: Author | None = _ALICE,
) -> ContextRequest:
    return ContextRequest(
        familiar_id="aria",
        channel_id=100,
        guild_id=1,
        author=author,
        utterance=utterance,
        modality=Modality.text,
        budget_tokens=2048,
        deadline_s=10.0,
    )


def _write_aliases(store: MemoryStore, mapping: dict[str, str]) -> None:
    store.write_file("people/_aliases.json", json.dumps(mapping))


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


# ---------------------------------------------------------------------------
# Speaker file — always loaded by canonical slug
# ---------------------------------------------------------------------------


class TestSpeakerFile:
    def test_speaker_file_included_when_present(self, store: MemoryStore) -> None:
        store.write_file("people/discord-1.md", "Alice likes ska.")

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
        store.write_file("people/discord-1.md", "A note about Alice.")

        contributions = lookup(
            store, _request(utterance="what's the weather?", author=_ALICE)
        )

        assert len(contributions) == 1
        assert "A note about Alice" in contributions[0].text

    def test_no_speaker_file_no_contribution(self, store: MemoryStore) -> None:
        contributions = lookup(store, _request(utterance="hello"))
        assert contributions == []

    def test_none_author_no_speaker_file(self, store: MemoryStore) -> None:
        """A null author (system-generated event) skips the speaker pass."""
        store.write_file("people/discord-1.md", "x")
        contributions = lookup(store, _request(utterance="hello", author=None))
        assert contributions == []

    def test_speaker_slug_is_canonical_key(self, store: MemoryStore) -> None:
        """File name comes from Author.slug — platform + user_id, not display name.

        A display-name-based file would miss here; the canonical slug
        guarantees the round-trip.
        """
        store.write_file("people/discord-1.md", "MJ notes.")
        contributions = lookup(store, _request(utterance="hi", author=_ALICE))
        assert len(contributions) == 1
        assert "MJ notes" in contributions[0].text


# ---------------------------------------------------------------------------
# Mentioned names — resolved via alias index
# ---------------------------------------------------------------------------


class TestMentionedNames:
    def test_capitalized_mention_loads_via_alias_index(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/discord-2.md", "Bob the Builder.")
        _write_aliases(store, {"bob": "discord-2"})
        contributions = lookup(
            store, _request(utterance="tell me about Bob", author=None)
        )
        assert len(contributions) == 1
        assert "Bob the Builder" in contributions[0].text

    def test_lowercase_mention_loads_via_alias_index(self, store: MemoryStore) -> None:
        """Tokens in the utterance are looked up in the alias index."""
        store.write_file("people/discord-2.md", "Bob notes.")
        _write_aliases(store, {"bob": "discord-2"})
        contributions = lookup(store, _request(utterance="where is bob?", author=None))
        assert len(contributions) == 1
        assert "Bob notes" in contributions[0].text

    def test_alias_resolves_nickname_to_canonical_slug(
        self, store: MemoryStore
    ) -> None:
        """Nickname → slug via alias index. Recall correctness floor."""
        store.write_file("people/discord-2.md", "Robert, aka Bob.")
        _write_aliases(store, {"bob": "discord-2", "rob": "discord-2"})
        contributions = lookup(
            store, _request(utterance="tell me about Rob", author=None)
        )
        assert len(contributions) == 1
        assert "Robert" in contributions[0].text

    def test_multiword_alias_matches_phrase(self, store: MemoryStore) -> None:
        store.write_file("people/discord-3.md", "a handyman")
        _write_aliases(store, {"bob the builder": "discord-3"})
        contributions = lookup(
            store,
            _request(utterance="where is bob the builder?", author=None),
        )
        assert len(contributions) == 1
        assert "handyman" in contributions[0].text

    def test_no_match_no_contribution(self, store: MemoryStore) -> None:
        store.write_file("people/discord-1.md", "A")
        store.write_file("people/discord-2.md", "B")
        _write_aliases(store, {"alice": "discord-1", "bob": "discord-2"})
        contributions = lookup(
            store, _request(utterance="what's for lunch?", author=None)
        )
        assert contributions == []

    def test_speaker_first_then_mentioned(self, store: MemoryStore) -> None:
        store.write_file("people/discord-1.md", "A")
        store.write_file("people/discord-2.md", "B")
        _write_aliases(store, {"alice": "discord-1", "bob": "discord-2"})
        contributions = lookup(store, _request(utterance="hi Bob", author=_ALICE))
        assert len(contributions) == 2
        assert contributions[0].text.strip() == "A"  # speaker first
        assert contributions[1].text.strip() == "B"

    def test_mention_dedupes_with_speaker(self, store: MemoryStore) -> None:
        store.write_file("people/discord-1.md", "A")
        _write_aliases(store, {"alice": "discord-1"})
        # speaker mentions themselves — still only one contribution
        contributions = lookup(
            store, _request(utterance="hi, I'm Alice", author=_ALICE)
        )
        assert len(contributions) == 1

    def test_missing_alias_index_silently_skips_mentions(
        self, store: MemoryStore
    ) -> None:
        """No alias index file → mentions pass returns nothing (not a crash)."""
        store.write_file("people/discord-2.md", "Bob.")
        contributions = lookup(store, _request(utterance="about Bob", author=None))
        assert contributions == []

    def test_malformed_alias_index_silently_skips_mentions(
        self, store: MemoryStore
    ) -> None:
        store.write_file("people/discord-2.md", "Bob.")
        store.write_file("people/_aliases.json", "{not valid json")
        contributions = lookup(store, _request(utterance="about Bob", author=None))
        assert contributions == []


# ---------------------------------------------------------------------------
# Budget handling
# ---------------------------------------------------------------------------


class TestBudget:
    def test_oversize_file_truncated(self, store: MemoryStore) -> None:
        long_text = "word " * 2000  # ~2500 tokens — well over per-file cap
        store.write_file("people/discord-1.md", long_text)
        contributions = lookup(store, _request())
        assert len(contributions) == 1
        # default per-file cap is 800 tokens; allow a little slack
        assert contributions[0].estimated_tokens <= 900

    def test_content_cap_drops_later_files(self, store: MemoryStore) -> None:
        """Budget overflow drops utterance-order tail, keeps speaker."""
        long_text = "word " * 1000  # truncates to per-file cap (~800 tokens)
        store.write_file("people/discord-1.md", long_text)
        store.write_file("people/discord-2.md", long_text)
        _write_aliases(store, {"bob": "discord-2"})

        contributions = lookup(
            store,
            _request(utterance="hi Bob", author=_ALICE),
            content_cap_tokens=900,  # leaves room for exactly one file
        )

        # speaker (Alice) wins; Bob dropped
        assert len(contributions) == 1
