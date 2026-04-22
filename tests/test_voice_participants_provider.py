"""Red-first tests for VoiceParticipantsProvider.

Surfaces voice-channel membership (`ContextRequest.voice_participants`)
as a ``Layer.author_note`` contribution so the LLM knows who else is
present on the call. Inert for text turns and when no participants
are supplied.
"""

from __future__ import annotations

from typing import Any

import pytest

from familiar_connect.context.protocols import ContextProvider
from familiar_connect.context.providers.voice_participants import (
    VOICE_PARTICIPANTS_PRIORITY,
    VoiceParticipantsProvider,
)
from familiar_connect.context.types import (
    ContextRequest,
    Layer,
    Modality,
)
from familiar_connect.identity import Author

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")
_BOB = Author(platform="discord", user_id="2", username="bob", display_name="Bob")
_CAROL = Author(platform="discord", user_id="3", username="carol", display_name="Carol")


def _make_request(**overrides: Any) -> ContextRequest:  # noqa: ANN401
    defaults: dict[str, Any] = {
        "familiar_id": "aria",
        "channel_id": 100,
        "guild_id": 1,
        "author": _ALICE,
        "utterance": "hello",
        "modality": Modality.voice,
        "budget_tokens": 2048,
        "deadline_s": 5.0,
    }
    defaults.update(overrides)
    return ContextRequest(**defaults)


class TestConstructionAndProtocol:
    def test_has_id_and_deadline(self) -> None:
        provider = VoiceParticipantsProvider()
        assert provider.id == "voice_participants"
        assert provider.deadline_s > 0

    def test_conforms_to_context_provider_protocol(self) -> None:
        assert isinstance(VoiceParticipantsProvider(), ContextProvider)


class TestGating:
    @pytest.mark.asyncio
    async def test_text_modality_returns_nothing(self) -> None:
        provider = VoiceParticipantsProvider()
        req = _make_request(
            modality=Modality.text,
            voice_participants=(_ALICE, _BOB),
        )
        assert await provider.contribute(req) == []

    @pytest.mark.asyncio
    async def test_voice_with_no_participants_returns_nothing(self) -> None:
        provider = VoiceParticipantsProvider()
        req = _make_request(voice_participants=())
        assert await provider.contribute(req) == []


class TestEmitsContribution:
    @pytest.mark.asyncio
    async def test_single_participant_mentioned(self) -> None:
        provider = VoiceParticipantsProvider()
        req = _make_request(voice_participants=(_ALICE,))
        (c,) = await provider.contribute(req)
        assert c.layer is Layer.author_note
        assert c.priority == VOICE_PARTICIPANTS_PRIORITY
        assert "Alice" in c.text

    @pytest.mark.asyncio
    async def test_multiple_participants_listed(self) -> None:
        provider = VoiceParticipantsProvider()
        req = _make_request(voice_participants=(_ALICE, _BOB, _CAROL))
        (c,) = await provider.contribute(req)
        assert "Alice" in c.text
        assert "Bob" in c.text
        assert "Carol" in c.text

    @pytest.mark.asyncio
    async def test_participant_order_preserved(self) -> None:
        """Caller controls ordering; provider does not re-sort."""
        provider = VoiceParticipantsProvider()
        req = _make_request(voice_participants=(_CAROL, _ALICE, _BOB))
        (c,) = await provider.contribute(req)
        assert c.text.index("Carol") < c.text.index("Alice") < c.text.index("Bob")

    @pytest.mark.asyncio
    async def test_source_tag_set(self) -> None:
        provider = VoiceParticipantsProvider()
        req = _make_request(voice_participants=(_ALICE,))
        (c,) = await provider.contribute(req)
        assert c.source == "voice_participants"

    @pytest.mark.asyncio
    async def test_estimated_tokens_positive(self) -> None:
        provider = VoiceParticipantsProvider()
        req = _make_request(voice_participants=(_ALICE, _BOB))
        (c,) = await provider.contribute(req)
        assert c.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_deduplicates_same_canonical_key(self) -> None:
        """Same user appearing twice (e.g. alias duplicate) renders once."""
        provider = VoiceParticipantsProvider()
        alice_dupe = Author(
            platform="discord",
            user_id="1",
            username="alice",
            display_name="Alice",
        )
        req = _make_request(voice_participants=(_ALICE, alice_dupe, _BOB))
        (c,) = await provider.contribute(req)
        # Alice appears only once despite being supplied twice.
        assert c.text.count("Alice") == 1
        assert "Bob" in c.text
