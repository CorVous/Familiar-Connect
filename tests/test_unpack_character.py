"""Red-first tests for the character-card unpacker.

Covers familiar_connect.memory.unpack_character, which doesn't exist
yet. Per future-features/context-management.md step 4: on familiar
creation, read the loaded CharacterCard and write one Markdown file
per non-empty field into the MemoryStore at ``self/<field>.md``. Empty
fields are skipped (no empty file on disk). Idempotent: re-unpacking
the same card is a no-op; re-unpacking a different card requires
``overwrite=True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.character import CharacterCard
from familiar_connect.memory.store import MemoryStore
from familiar_connect.memory.unpack_character import (
    CharacterUnpackError,
    unpack_character,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Return a fresh MemoryStore for each test."""
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def full_card() -> CharacterCard:
    """Return a CharacterCard with every field populated."""
    return CharacterCard(
        name="Aria",
        description="A small cat-shaped familiar with a fondness for tea.",
        personality="Curious, sly, occasionally bossy.",
        scenario="Late evening in the study, fire crackling.",
        first_mes="*peers up from a teacup* About time you got here.",
        mes_example="<START>\n{{user}}: Hi.\n{{char}}: *blinks slowly*",
        system_prompt="Be brief, charming, and slightly aloof.",
        post_history_instructions="Reread the last user message before replying.",
        creator_notes="Built for after-dinner conversation.",
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestUnpackHappyPaths:
    def test_writes_non_empty_fields_into_self(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        written = unpack_character(store, full_card)
        # Every populated field becomes a self/<field>.md path.
        expected = {
            "self/name.md",
            "self/description.md",
            "self/personality.md",
            "self/scenario.md",
            "self/first_mes.md",
            "self/mes_example.md",
            "self/system_prompt.md",
            "self/post_history_instructions.md",
            "self/creator_notes.md",
        }
        assert set(written) == expected

    def test_round_trip_each_field(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        unpack_character(store, full_card)
        assert store.read_file("self/name.md") == "Aria"
        assert (
            store.read_file("self/description.md")
            == "A small cat-shaped familiar with a fondness for tea."
        )
        assert (
            store.read_file("self/personality.md")
            == "Curious, sly, occasionally bossy."
        )
        assert (
            store.read_file("self/scenario.md")
            == "Late evening in the study, fire crackling."
        )

    def test_skips_empty_fields(self, store: MemoryStore) -> None:
        sparse = CharacterCard(
            name="Quiet",
            description="Just a description.",
            personality="",
            scenario="",
            first_mes="",
            mes_example="",
            system_prompt="",
            post_history_instructions="",
            creator_notes="",
        )
        written = unpack_character(store, sparse)
        assert set(written) == {"self/name.md", "self/description.md"}
        # Skipped fields don't exist on disk.
        for skipped in (
            "self/personality.md",
            "self/scenario.md",
            "self/first_mes.md",
            "self/mes_example.md",
            "self/system_prompt.md",
            "self/post_history_instructions.md",
            "self/creator_notes.md",
        ):
            with pytest.raises(Exception):  # noqa: B017, PT011
                store.read_file(skipped)

    def test_records_audit_entries_for_each_write(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        unpack_character(store, full_card)
        sources = {e.source for e in store.audit_entries}
        # Audit source is identifiable as the unpacker.
        assert all(s.startswith("character_card_unpacker") for s in sources)
        # Number of audit entries == number of files written.
        assert len(store.audit_entries) == 9


# ---------------------------------------------------------------------------
# Idempotency and overwrite
# ---------------------------------------------------------------------------


class TestUnpackIdempotency:
    def test_reunpack_same_card_is_noop(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        unpack_character(store, full_card)
        first_audit_count = len(store.audit_entries)

        # Second unpack of the exact same card.
        written = unpack_character(store, full_card)

        assert written == []
        assert len(store.audit_entries) == first_audit_count

    def test_reunpack_different_card_without_overwrite_raises(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        unpack_character(store, full_card)

        modified = CharacterCard(
            name="Aria",
            description="A subtly different description.",
            personality=full_card.personality,
            scenario=full_card.scenario,
            first_mes=full_card.first_mes,
            mes_example=full_card.mes_example,
            system_prompt=full_card.system_prompt,
            post_history_instructions=full_card.post_history_instructions,
            creator_notes=full_card.creator_notes,
        )

        with pytest.raises(CharacterUnpackError):
            unpack_character(store, modified)

        # The original on-disk content is preserved.
        assert store.read_file("self/description.md") == full_card.description

    def test_reunpack_with_overwrite_replaces_changed_fields(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        unpack_character(store, full_card)

        modified = CharacterCard(
            name=full_card.name,
            description="Replaced description.",
            personality=full_card.personality,
            scenario="Replaced scenario.",
            first_mes=full_card.first_mes,
            mes_example=full_card.mes_example,
            system_prompt=full_card.system_prompt,
            post_history_instructions=full_card.post_history_instructions,
            creator_notes=full_card.creator_notes,
        )

        written = unpack_character(store, modified, overwrite=True)

        # Only the *changed* fields were rewritten.
        assert set(written) == {"self/description.md", "self/scenario.md"}
        assert store.read_file("self/description.md") == "Replaced description."
        assert store.read_file("self/scenario.md") == "Replaced scenario."
        # Unchanged fields still match the original card.
        assert store.read_file("self/personality.md") == full_card.personality

    def test_overwrite_can_remove_a_field(
        self, store: MemoryStore, full_card: CharacterCard
    ) -> None:
        """A field that becomes empty is deleted from disk under overwrite."""
        unpack_character(store, full_card)

        cleared = CharacterCard(
            name=full_card.name,
            description=full_card.description,
            personality="",  # cleared
            scenario=full_card.scenario,
            first_mes=full_card.first_mes,
            mes_example=full_card.mes_example,
            system_prompt=full_card.system_prompt,
            post_history_instructions=full_card.post_history_instructions,
            creator_notes=full_card.creator_notes,
        )

        unpack_character(store, cleared, overwrite=True)

        with pytest.raises(Exception):  # noqa: B017, PT011
            store.read_file("self/personality.md")
        # Other fields still intact.
        assert store.read_file("self/description.md") == full_card.description
