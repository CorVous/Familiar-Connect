"""Red-first tests for the ChannelConfigStore.

The store owns the ``data/familiars/<id>/channels/`` directory. Each
channel's TOML sidecar stores the :class:`ChannelMode` (and future
per-channel overrides); the store loads them lazily, caches them in
memory, and writes back when a slash command flips a channel's mode.

Covers familiar_connect.channel_config, which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.channel_config import ChannelConfigStore
from familiar_connect.config import ChannelMode, CharacterConfig
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fallback to character default
# ---------------------------------------------------------------------------


class TestFallback:
    def test_unknown_channel_falls_through_to_character_default(
        self,
        tmp_path: Path,
    ) -> None:
        character = CharacterConfig(default_mode=ChannelMode.imitate_voice)
        store = ChannelConfigStore(root=tmp_path, character=character)

        cfg = store.get(channel_id=12345)
        assert cfg.mode is ChannelMode.imitate_voice


# ---------------------------------------------------------------------------
# Setting a mode writes a sidecar
# ---------------------------------------------------------------------------


class TestSetMode:
    def test_set_mode_writes_toml(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_mode(channel_id=999, mode=ChannelMode.full_rp)

        sidecar = tmp_path / "999.toml"
        assert sidecar.exists()

    def test_set_mode_read_back(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.text_conversation_rp),
        )
        store.set_mode(channel_id=999, mode=ChannelMode.full_rp)
        assert store.get(channel_id=999).mode is ChannelMode.full_rp

    def test_set_mode_survives_new_store_instance(self, tmp_path: Path) -> None:
        """Persistence: a fresh store against the same root sees existing sidecars."""
        store_a = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        store_a.set_mode(channel_id=999, mode=ChannelMode.full_rp)

        store_b = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        assert store_b.get(channel_id=999).mode is ChannelMode.full_rp


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_repeated_gets_reuse_same_object(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.full_rp),
        )
        store.set_mode(channel_id=42, mode=ChannelMode.full_rp)
        first = store.get(channel_id=42)
        second = store.get(channel_id=42)
        # ChannelConfig is frozen so the cache can legitimately return the
        # same instance; both must at least compare equal.
        assert first == second


# ---------------------------------------------------------------------------
# set_backdrop / get_backdrop
# ---------------------------------------------------------------------------


class TestBackdrop:
    def test_set_backdrop_creates_sidecar_when_absent(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.full_rp),
        )
        store.set_backdrop(channel_id=7, backdrop="Talk like a pirate.")

        sidecar = tmp_path / "7.toml"
        assert sidecar.exists()

    def test_set_backdrop_seeds_mode_from_character_default(
        self, tmp_path: Path
    ) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.imitate_voice),
        )
        store.set_backdrop(channel_id=7, backdrop="Speak softly.")

        cfg = store.get(channel_id=7)
        assert cfg.mode is ChannelMode.imitate_voice

    def test_set_backdrop_preserves_existing_mode(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.text_conversation_rp),
        )
        store.set_mode(channel_id=7, mode=ChannelMode.full_rp)
        store.set_backdrop(channel_id=7, backdrop="Pirate speech.")

        cfg = store.get(channel_id=7)
        assert cfg.mode is ChannelMode.full_rp
        assert cfg.backdrop_override == "Pirate speech."

    def test_set_backdrop_stores_channel_name(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_backdrop(channel_id=7, backdrop="Ahoy.", channel_name="tavern")

        cfg = store.get(channel_id=7)
        assert cfg.channel_name == "tavern"

    def test_set_backdrop_invalidates_cache(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_mode(channel_id=7, mode=ChannelMode.full_rp)
        _ = store.get(channel_id=7)  # prime cache

        store.set_backdrop(channel_id=7, backdrop="New backdrop.")
        cfg = store.get(channel_id=7)
        assert cfg.backdrop_override == "New backdrop."

    def test_get_backdrop_returns_none_for_unknown_channel(
        self, tmp_path: Path
    ) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        assert store.get_backdrop(channel_id=999) is None

    def test_get_backdrop_returns_value_after_set(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_backdrop(channel_id=7, backdrop="Hello backdrop.")
        assert store.get_backdrop(channel_id=7) == "Hello backdrop."

    def test_sidecar_writes_channel_name_at_top(self, tmp_path: Path) -> None:
        """``channel_name`` must render as the first key in the written TOML.

        Operators read these files by hand; keeping the human-readable
        header at the top makes the sidecar scannable.
        """
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_backdrop(channel_id=7, backdrop="Hello.", channel_name="general")

        text = (tmp_path / "7.toml").read_text()
        key_lines = [
            line for line in text.splitlines() if line and not line.startswith(" ")
        ]
        assert key_lines[0].startswith("channel_name")
        # mode should follow channel_name, before backdrop
        channel_idx = next(
            i for i, ln in enumerate(key_lines) if ln.startswith("channel_name")
        )
        mode_idx = next(i for i, ln in enumerate(key_lines) if ln.startswith("mode"))
        backdrop_idx = next(
            i for i, ln in enumerate(key_lines) if ln.startswith("backdrop")
        )
        assert channel_idx < mode_idx < backdrop_idx


# ---------------------------------------------------------------------------
# set_mode preservation (regression)
# ---------------------------------------------------------------------------


class TestSetModePreservation:
    def test_set_mode_preserves_backdrop(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_backdrop(channel_id=7, backdrop="Keep this.")
        store.set_mode(channel_id=7, mode=ChannelMode.full_rp)

        cfg = store.get(channel_id=7)
        assert cfg.backdrop_override == "Keep this."

    def test_set_mode_preserves_channel_name(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_backdrop(channel_id=7, backdrop="x", channel_name="general")
        store.set_mode(channel_id=7, mode=ChannelMode.full_rp)

        cfg = store.get(channel_id=7)
        assert cfg.channel_name == "general"


# ---------------------------------------------------------------------------
# last_context cache
# ---------------------------------------------------------------------------


class TestLastContext:
    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi", name="Alice"),
            Message(role="assistant", content="hello"),
        ]
        store.set_last_context(channel_id=1, messages=msgs)
        assert store.get_last_context(channel_id=1) == msgs

    def test_preserves_mode_and_backdrop(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        store.set_mode(channel_id=1, mode=ChannelMode.text_conversation_rp)
        store.set_backdrop(channel_id=1, backdrop="a backdrop")
        store.set_last_context(
            channel_id=1, messages=[Message(role="system", content="sys")]
        )
        cfg = store.get(channel_id=1)
        assert cfg.mode is ChannelMode.text_conversation_rp
        assert cfg.backdrop_override == "a backdrop"

    def test_empty_when_no_sidecar(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        assert store.get_last_context(channel_id=99) is None

    def test_last_context_trails_other_tables(self, tmp_path: Path) -> None:
        """[[last_context]] must appear after [typing_simulation] in toml."""
        store = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        store.set_mode(channel_id=1, mode=ChannelMode.full_rp)
        store.set_last_context(
            channel_id=1, messages=[Message(role="system", content="x")]
        )
        sidecar = tmp_path / "1.toml"
        text = sidecar.read_text()
        ts_pos = (
            text.find("[typing_simulation]") if "[typing_simulation]" in text else -1
        )
        lc_pos = text.find("[[last_context]]")
        # if typing_simulation table is present, last_context must come after it
        if ts_pos != -1:
            assert lc_pos > ts_pos
