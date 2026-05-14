"""Tests for ``familiar-connect init`` — narrative → authored canon."""

from __future__ import annotations

import argparse
import asyncio
import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from familiar_connect.cli import create_parser
from familiar_connect.commands.init import (
    InitError,
    _build_plan,
    _coerce_character_overlay,
    _coerce_lorebook_entries,
    _parse_json_object,
    _read_narrative,
    _write_plan,
    init,
)
from familiar_connect.context.assembler import AssemblyContext
from familiar_connect.context.layers import LorebookLayer
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, Message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _ScriptedLLM(LLMClient):
    """LLM stub: returns canned replies in order; records call inputs."""

    def __init__(self, *, replies: list[str]) -> None:
        super().__init__(api_key="k", model="m")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content="")
        return Message(role="assistant", content=self._replies.pop(0))

    async def chat_stream(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = await self.chat(messages)
        yield reply.content

    async def close(self) -> None:
        return None


def _seed_narrative(root: Path) -> Path:
    src = root / "narrative"
    src.mkdir()
    (src / "01-intro.md").write_text(
        "# Aria\n\nAria runs the Lantern Press in Old Town.\n",
        encoding="utf-8",
    )
    (src / "02-people.md").write_text(
        "## Bram\n\nGruff bookbinder, drinks his tea cold.\n",
        encoding="utf-8",
    )
    return src


def _lorebook_reply(entries: list[dict[str, object]]) -> str:
    return json.dumps({"entries": entries})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestReadNarrative:
    def test_concatenates_with_headers(self, tmp_path: Path) -> None:
        src = _seed_narrative(tmp_path)
        text = _read_narrative(src)
        assert "## 01-intro.md" in text
        assert "## 02-people.md" in text
        assert "Lantern Press" in text
        assert "Bram" in text

    def test_sort_is_stable(self, tmp_path: Path) -> None:
        src = tmp_path / "n"
        src.mkdir()
        (src / "b.md").write_text("B", encoding="utf-8")
        (src / "a.md").write_text("A", encoding="utf-8")
        text = _read_narrative(src)
        assert text.index("a.md") < text.index("b.md")


class TestParseJsonObject:
    def test_parses_plain_json(self) -> None:
        out = _parse_json_object('{"display_tz": "UTC"}', artifact="x")
        assert out == {"display_tz": "UTC"}

    def test_strips_code_fences(self) -> None:
        out = _parse_json_object(
            '```json\n{"display_tz": "Europe/Berlin"}\n```', artifact="x"
        )
        assert out == {"display_tz": "Europe/Berlin"}

    def test_rejects_non_object(self) -> None:
        with pytest.raises(InitError, match=r"character\.toml"):
            _parse_json_object("[1, 2]", artifact="character.toml")

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(InitError, match=r"lorebook\.toml"):
            _parse_json_object("{not json", artifact="lorebook.toml")


class TestCoerceCharacterOverlay:
    def test_keeps_known_fields(self) -> None:
        out = _coerce_character_overlay({
            "display_tz": "Europe/Berlin",
            "aliases": ["Aria", "the keeper"],
        })
        assert out == {
            "display_tz": "Europe/Berlin",
            "aliases": ["Aria", "the keeper"],
        }

    def test_drops_unknown_fields(self) -> None:
        out = _coerce_character_overlay({"display_tz": "UTC", "evil": True})
        assert "evil" not in out

    def test_drops_empty_aliases(self) -> None:
        out = _coerce_character_overlay({"aliases": ["", "  "]})
        assert "aliases" not in out


class TestCoerceLorebookEntries:
    def test_keeps_valid_entries(self) -> None:
        out = _coerce_lorebook_entries({
            "entries": [
                {
                    "keys": ["Lantern Press"],
                    "content": "Aria's bookshop in Old Town.",
                    "priority": 80,
                }
            ]
        })
        assert len(out) == 1
        assert out[0]["keys"] == ["Lantern Press"]
        assert out[0]["priority"] == 80

    def test_drops_entries_missing_keys(self) -> None:
        out = _coerce_lorebook_entries({"entries": [{"keys": [], "content": "x"}]})
        assert out == []

    def test_clamps_priority(self) -> None:
        out = _coerce_lorebook_entries({
            "entries": [
                {"keys": ["a"], "content": "x", "priority": 9999},
                {"keys": ["b"], "content": "y", "priority": -50},
            ]
        })
        assert out[0]["priority"] == 100
        assert out[1]["priority"] == 0

    def test_returns_empty_on_missing_entries(self) -> None:
        assert _coerce_lorebook_entries({}) == []


# ---------------------------------------------------------------------------
# End-to-end plan building
# ---------------------------------------------------------------------------


class TestBuildPlan:
    def test_three_artifacts_in_order(self) -> None:
        llm = _ScriptedLLM(
            replies=[
                "Aria is a careful, soft-spoken bookbinder.",
                json.dumps({"display_tz": "Europe/Berlin", "aliases": ["A"]}),
                _lorebook_reply([
                    {
                        "keys": ["Lantern Press"],
                        "content": "Aria's bookshop in Old Town.",
                        "priority": 80,
                    }
                ]),
            ]
        )
        plan = asyncio.run(_build_plan(llm, "Some narrative."))
        assert set(plan) == {"character.md", "character.toml", "lorebook.toml"}
        assert "soft-spoken" in plan["character.md"]
        # character.toml parses back as TOML and carries the overlay
        toml_data = tomllib.loads(plan["character.toml"])
        assert toml_data["display_tz"] == "Europe/Berlin"
        # lorebook.toml parses back and has the entry
        lore = tomllib.loads(plan["lorebook.toml"])
        assert lore["entries"][0]["keys"] == ["Lantern Press"]
        # later calls saw earlier outputs
        assert any("soft-spoken" in m.content for m in llm.calls[1])

    def test_empty_persona_raises(self) -> None:
        llm = _ScriptedLLM(replies=["", "{}", _lorebook_reply([])])
        with pytest.raises(InitError, match=r"character\.md"):
            asyncio.run(_build_plan(llm, "x"))


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


class TestWritePlan:
    def test_writes_only_non_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "aria"
        _write_plan(
            target,
            {
                "character.md": "hi\n",
                "character.toml": "",
                "lorebook.toml": "entries = []\n",
            },
        )
        assert (target / "character.md").read_text(encoding="utf-8") == "hi\n"
        assert not (target / "character.toml").exists()
        assert (target / "lorebook.toml").exists()

    def test_writes_are_atomic_on_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "aria"
        target.mkdir()
        (target / "character.md").write_text("old", encoding="utf-8")
        _write_plan(target, {"character.md": "new\n"})
        assert (target / "character.md").read_text(encoding="utf-8") == "new\n"


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------


class TestCli:
    def test_subcommand_registered(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["init", "aria", "--from", "n"])
        assert args.command == "init"
        assert args.familiar_id == "aria"
        assert args.source == Path("n")

    def test_missing_source_dir_exits_nonzero(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            familiar_id="aria",
            source=tmp_path / "nope",
            force=False,
            dry_run=False,
        )
        assert init(args) == 1

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path) -> None:
        src = _seed_narrative(tmp_path)
        target = tmp_path / "data" / "familiars" / "aria"
        target.mkdir(parents=True)
        (target / "character.md").write_text("existing", encoding="utf-8")
        args = argparse.Namespace(
            familiar_id="aria",
            source=src,
            force=False,
            dry_run=False,
        )
        with patch(
            "familiar_connect.commands.init._DEFAULT_FAMILIARS_ROOT",
            tmp_path / "data" / "familiars",
        ):
            assert init(args) == 1
        # untouched
        assert (target / "character.md").read_text(encoding="utf-8") == "existing"

    def test_refuses_reserved_default_id(self, tmp_path: Path) -> None:
        src = _seed_narrative(tmp_path)
        args = argparse.Namespace(
            familiar_id="_default",
            source=src,
            force=False,
            dry_run=False,
        )
        assert init(args) == 1

    def test_dry_run_does_not_write(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = _seed_narrative(tmp_path)
        stub = _ScriptedLLM(
            replies=[
                "Persona text.",
                "{}",
                _lorebook_reply([
                    {"keys": ["Lantern"], "content": "the shop", "priority": 50}
                ]),
            ]
        )
        args = argparse.Namespace(
            familiar_id="aria",
            source=src,
            force=False,
            dry_run=True,
        )
        with (
            patch(
                "familiar_connect.commands.init._DEFAULT_FAMILIARS_ROOT",
                tmp_path / "data" / "familiars",
            ),
            patch(
                "familiar_connect.commands.init._load_background_llm",
                return_value=stub,
            ),
        ):
            assert init(args) == 0
        assert not (tmp_path / "data" / "familiars" / "aria").exists()
        out = capsys.readouterr().out
        assert "character.md" in out
        assert "lorebook.toml" in out

    def test_real_run_writes_three_files(self, tmp_path: Path) -> None:
        src = _seed_narrative(tmp_path)
        stub = _ScriptedLLM(
            replies=[
                "Aria is a careful, soft-spoken bookbinder.",
                json.dumps({"display_tz": "Europe/Berlin"}),
                _lorebook_reply([
                    {
                        "keys": ["Lantern Press"],
                        "content": "Aria's bookshop in Old Town.",
                        "priority": 80,
                    }
                ]),
            ]
        )
        args = argparse.Namespace(
            familiar_id="aria",
            source=src,
            force=False,
            dry_run=False,
        )
        target_root = tmp_path / "data" / "familiars"
        with (
            patch(
                "familiar_connect.commands.init._DEFAULT_FAMILIARS_ROOT",
                target_root,
            ),
            patch(
                "familiar_connect.commands.init._load_background_llm",
                return_value=stub,
            ),
        ):
            assert init(args) == 0
        target = target_root / "aria"
        assert (target / "character.md").read_text(encoding="utf-8")
        toml_data = tomllib.loads(
            (target / "character.toml").read_text(encoding="utf-8")
        )
        assert toml_data["display_tz"] == "Europe/Berlin"
        lore = tomllib.loads((target / "lorebook.toml").read_text(encoding="utf-8"))
        assert lore["entries"][0]["keys"] == ["Lantern Press"]
        # critically: no history.db, no facts, no dossiers
        assert not (target / "history.db").exists()


# ---------------------------------------------------------------------------
# Integration: lorebook layer picks up generated entries
# ---------------------------------------------------------------------------


class TestLorebookIntegration:
    def test_generated_lorebook_activates_in_assembly(self, tmp_path: Path) -> None:
        """End-to-end: a turn mentioning a generated key surfaces the entry."""
        src = _seed_narrative(tmp_path)
        stub = _ScriptedLLM(
            replies=[
                "Aria persona.",
                "{}",
                _lorebook_reply([
                    {
                        "keys": ["Lantern Press"],
                        "content": "Aria's bookshop in Old Town.",
                        "priority": 80,
                    }
                ]),
            ]
        )
        args = argparse.Namespace(
            familiar_id="aria",
            source=src,
            force=False,
            dry_run=False,
        )
        target_root = tmp_path / "data" / "familiars"
        with (
            patch(
                "familiar_connect.commands.init._DEFAULT_FAMILIARS_ROOT",
                target_root,
            ),
            patch(
                "familiar_connect.commands.init._load_background_llm",
                return_value=stub,
            ),
        ):
            assert init(args) == 0

        # wire a HistoryStore + LorebookLayer, seed one turn mentioning a key
        store = HistoryStore(tmp_path / "history.db")
        store.append_turn(
            familiar_id="aria",
            channel_id=1,
            role="user",
            content="Have you been by the Lantern Press lately?",
        )
        async_store = AsyncHistoryStore(store)
        layer = LorebookLayer(
            store=async_store,
            path=target_root / "aria" / "lorebook.toml",
            recent_window=10,
            max_entries=10,
            max_tokens=1000,
        )
        ctx = AssemblyContext(familiar_id="aria", channel_id=1, viewer_mode="text")
        text = asyncio.run(layer.build(ctx))
        store.close()
        assert "Old Town" in text
