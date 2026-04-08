"""Red-first tests for CharacterProvider.

Step 5 of future-features/context-management.md. Reads ``self/*.md``
from the familiar's MemoryStore and emits one Contribution per
non-empty file at high priority on Layer.character. Always on (no
modality toggle); a familiar with no character description is a
degenerate state but the provider must still return cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.character import (
    CHARACTER_PRIORITY,
    CharacterProvider,
)
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Return a fresh MemoryStore for each test."""
    return MemoryStore(tmp_path / "memory")


def _make_request() -> ContextRequest:
    return ContextRequest(
        guild_id=1,
        familiar_id="aria",
        channel_id=100,
        speaker="Alice",
        utterance="hello",
        modality=Modality.text,
        budget_tokens=2048,
        deadline_s=5.0,
    )


# ---------------------------------------------------------------------------
# Construction and protocol conformance
# ---------------------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_has_id_and_deadline(self, store: MemoryStore) -> None:
        provider = CharacterProvider(store)
        assert provider.id == "character"
        assert provider.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self, store: MemoryStore) -> None:
        provider = CharacterProvider(store)
        assert isinstance(provider, ContextProvider)


# ---------------------------------------------------------------------------
# Reading self/ files
# ---------------------------------------------------------------------------


class TestReadsSelfFiles:
    @pytest.mark.asyncio
    async def test_empty_self_returns_no_contributions(
        self, store: MemoryStore
    ) -> None:
        provider = CharacterProvider(store)
        contributions = await provider.contribute(_make_request())
        assert contributions == []

    @pytest.mark.asyncio
    async def test_returns_one_contribution_per_self_file(
        self, store: MemoryStore
    ) -> None:
        store.write_file("self/description.md", "A small cat-shaped familiar.")
        store.write_file("self/personality.md", "Curious and sly.")
        store.write_file("self/scenario.md", "Late evening in the study.")

        provider = CharacterProvider(store)
        contributions = await provider.contribute(_make_request())

        assert len(contributions) == 3
        for c in contributions:
            assert isinstance(c, Contribution)
            assert c.layer is Layer.character
            assert c.priority == CHARACTER_PRIORITY

    @pytest.mark.asyncio
    async def test_contribution_text_matches_file_contents(
        self, store: MemoryStore
    ) -> None:
        store.write_file("self/description.md", "A small cat-shaped familiar.")

        provider = CharacterProvider(store)
        contributions = await provider.contribute(_make_request())

        assert len(contributions) == 1
        assert contributions[0].text == "A small cat-shaped familiar."

    @pytest.mark.asyncio
    async def test_source_identifies_field(self, store: MemoryStore) -> None:
        store.write_file("self/personality.md", "Sly.")

        provider = CharacterProvider(store)
        (c,) = await provider.contribute(_make_request())

        assert "personality" in c.source

    @pytest.mark.asyncio
    async def test_skips_empty_files(self, store: MemoryStore) -> None:
        store.write_file("self/description.md", "non-empty")
        store.write_file("self/personality.md", "")

        provider = CharacterProvider(store)
        contributions = await provider.contribute(_make_request())

        assert len(contributions) == 1
        assert contributions[0].text == "non-empty"

    @pytest.mark.asyncio
    async def test_skips_non_markdown_files(self, store: MemoryStore) -> None:
        store.write_file("self/description.md", "kept")
        store.write_file("self/notes.txt", "ignored")
        store.write_file("self/scratch.json", "{}")

        provider = CharacterProvider(store)
        contributions = await provider.contribute(_make_request())

        assert len(contributions) == 1
        assert contributions[0].text == "kept"

    @pytest.mark.asyncio
    async def test_skips_subdirectories_inside_self(self, store: MemoryStore) -> None:
        """``self/`` is a flat namespace; nested directories are not unpacked."""
        store.write_file("self/description.md", "kept")
        store.write_file("self/extras/extra.md", "ignored")

        provider = CharacterProvider(store)
        contributions = await provider.contribute(_make_request())

        rels = [c.source for c in contributions]
        assert any("description" in s for s in rels)
        assert not any("extra" in s for s in rels)

    @pytest.mark.asyncio
    async def test_estimated_tokens_is_set(self, store: MemoryStore) -> None:
        store.write_file("self/description.md", "x" * 40)
        provider = CharacterProvider(store)
        (c,) = await provider.contribute(_make_request())
        # Char-count heuristic puts 40 chars at ~10 tokens.
        assert c.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_returns_contributions_in_deterministic_order(
        self, store: MemoryStore
    ) -> None:
        store.write_file("self/description.md", "d")
        store.write_file("self/personality.md", "p")
        store.write_file("self/scenario.md", "s")

        provider = CharacterProvider(store)
        first = await provider.contribute(_make_request())
        second = await provider.contribute(_make_request())

        assert [c.source for c in first] == [c.source for c in second]
