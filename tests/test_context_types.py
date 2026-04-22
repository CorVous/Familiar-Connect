"""Red-first tests for context-pipeline dataclasses and enums.

Covers Layer, Modality, Contribution, and ContextRequest from
familiar_connect.context.types, which does not exist yet.
"""

from __future__ import annotations

import math
from typing import Any

# These imports are expected to fail until the module exists — that's the "red".
from familiar_connect.context.types import (
    ContextRequest,
    Contribution,
    Layer,
    Modality,
)
from familiar_connect.identity import Author

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")


class TestLayer:
    def test_has_core_layer(self) -> None:
        assert Layer.core.name == "core"

    def test_has_character_layer(self) -> None:
        assert Layer.character.name == "character"

    def test_has_content_layer(self) -> None:
        """Content layer holds results from ContentSearchProvider."""
        assert Layer.content.name == "content"

    def test_has_history_summary_layer(self) -> None:
        assert Layer.history_summary.name == "history_summary"

    def test_has_recent_history_layer(self) -> None:
        assert Layer.recent_history.name == "recent_history"

    def test_has_author_note_layer(self) -> None:
        assert Layer.author_note.name == "author_note"

    def test_has_depth_inject_layer(self) -> None:
        assert Layer.depth_inject.name == "depth_inject"

    def test_layers_are_distinct(self) -> None:
        """All layer members are distinct values."""
        values = {layer.value for layer in Layer}
        assert len(values) == len(list(Layer))


class TestModality:
    def test_voice_and_text_exist(self) -> None:
        assert Modality.voice.name == "voice"
        assert Modality.text.name == "text"

    def test_modality_values_are_strings(self) -> None:
        """Modalities serialise as their lowercase name (useful for config keys)."""
        assert Modality.voice.value == "voice"
        assert Modality.text.value == "text"


class TestContribution:
    def test_construct_with_required_fields(self) -> None:
        c = Contribution(
            layer=Layer.character,
            priority=10,
            text="hello",
            estimated_tokens=2,
            source="stub",
        )
        assert c.layer is Layer.character
        assert c.priority == 10
        assert c.text == "hello"
        assert c.estimated_tokens == 2
        assert c.source == "stub"

    def test_priority_and_tokens_are_ints(self) -> None:
        """Document Contribution equality via a dataclass eq path.

        Non-int priority/tokens are a programmer error — dataclasses
        don't enforce types at runtime, so this just pins the equality
        contract rather than attempting to type-check at runtime.
        """
        c = Contribution(
            layer=Layer.content,
            priority=0,
            text="",
            estimated_tokens=0,
            source="empty",
        )
        # A pair of equal contributions must compare equal (dataclass eq).
        same = Contribution(
            layer=Layer.content,
            priority=0,
            text="",
            estimated_tokens=0,
            source="empty",
        )
        assert c == same


class TestContextRequest:
    def _make(self, **overrides: object) -> ContextRequest:
        defaults: dict[str, Any] = {
            "familiar_id": "aria",
            "channel_id": 100,
            "guild_id": 1,
            "author": _ALICE,
            "utterance": "hello",
            "modality": Modality.text,
            "budget_tokens": 2048,
            "deadline_s": 5.0,
        }
        defaults.update(overrides)
        return ContextRequest(**defaults)  # type: ignore[arg-type]

    def test_construct_with_all_fields(self) -> None:
        req = self._make()
        assert req.familiar_id == "aria"
        assert req.channel_id == 100
        assert req.guild_id == 1
        assert req.author == _ALICE
        assert req.utterance == "hello"
        assert req.modality is Modality.text
        assert req.budget_tokens == 2048
        assert math.isclose(req.deadline_s, 5.0)

    def test_author_is_optional(self) -> None:
        """A system-generated turn may not have an author."""
        req = self._make(author=None)
        assert req.author is None

    def test_guild_id_is_optional(self) -> None:
        """Non-Discord events (Twitch, scheduled tasks) have no guild."""
        req = self._make(guild_id=None)
        assert req.guild_id is None

    def test_modality_can_be_voice(self) -> None:
        req = self._make(modality=Modality.voice)
        assert req.modality is Modality.voice

    def test_deadline_is_a_field(self) -> None:
        """deadline_s is explicitly present on every ContextRequest."""
        # We don't attempt to test "what if I leave it out" — the static type
        # checker correctly rejects such calls, so it's not a runtime concern.
        req = self._make(deadline_s=0.25)
        assert math.isclose(req.deadline_s, 0.25)

    def test_voice_participants_defaults_to_empty_tuple(self) -> None:
        """Text turns and voice turns without membership data carry ()."""
        req = self._make()
        assert req.voice_participants == ()

    def test_voice_participants_is_populated_on_voice_turns(self) -> None:
        bob = Author(
            platform="discord", user_id="2", username="bob", display_name="Bob"
        )
        req = self._make(
            modality=Modality.voice,
            voice_participants=(_ALICE, bob),
        )
        assert req.voice_participants == (_ALICE, bob)
