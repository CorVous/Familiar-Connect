"""Red-first tests for the post-session memory writer.

The MemoryWriter reads unsummarized turns from HistoryStore, calls a
cheap side-model to produce structured output, and writes session/people/
topic files into the MemoryStore.

Covers familiar_connect.memory.writer.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.memory.store import MemoryStore
from familiar_connect.memory.writer import (
    MemoryWriter,
    _parse_writer_output,
    _session_filename,
    _time_slot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.llm import Message

# Re-use the shared FakeLLMClient from conftest.
from tests.conftest import FakeLLMClient

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_FAMILIAR = "aria"

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")
_BOB = Author(platform="discord", user_id="2", username="bob", display_name="Bob")


def _make_stores(tmp_path: Path) -> tuple[MemoryStore, HistoryStore]:
    mem_root = tmp_path / "memory"
    mem_root.mkdir()
    return MemoryStore(mem_root), HistoryStore(tmp_path / "history.db")


def _seed_turns(store: HistoryStore, n: int) -> None:
    """Append *n* alternating user/assistant turns."""
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        author = _ALICE if role == "user" else None
        store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=100,
            role=role,
            content=f"turn {i}",
            author=author,
        )


_STRUCTURED_OUTPUT = """\
===SESSION_SUMMARY===
Alice and the familiar had a brief exchange about greetings.
===END_SESSION_SUMMARY===

===PEOPLE===
---FILE: discord-1.md---
# Alice

Alice is a friendly person who greeted the familiar.

## Impressions
- Seems warm and approachable

## Aliases
- Alice
- alice
---END_FILE---
===END_PEOPLE===

===TOPICS===
---FILE: greetings.md---
# Greetings

A conversation about saying hello.

## Notes
- Alice prefers casual greetings
---END_FILE---
===END_TOPICS==="""


# ---------------------------------------------------------------------------
# Output parser tests
# ---------------------------------------------------------------------------


class TestParseWriterOutput:
    def test_parses_all_sections(self) -> None:
        session, people, topics = _parse_writer_output(_STRUCTURED_OUTPUT)
        assert "Alice and the familiar" in session
        assert "discord-1.md" in people
        assert "# Alice" in people["discord-1.md"]
        assert "greetings.md" in topics
        assert "# Greetings" in topics["greetings.md"]

    def test_missing_session_returns_empty(self) -> None:
        text = "===PEOPLE===\n===END_PEOPLE===\n===TOPICS===\n===END_TOPICS==="
        session, people, topics = _parse_writer_output(text)
        assert not session
        assert people == {}
        assert topics == {}

    def test_empty_input(self) -> None:
        session, people, topics = _parse_writer_output("")
        assert not session
        assert people == {}
        assert topics == {}

    def test_multiple_people_files(self) -> None:
        text = """\
===SESSION_SUMMARY===
Summary
===END_SESSION_SUMMARY===

===PEOPLE===
---FILE: alice.md---
# Alice
Content about Alice
---END_FILE---
---FILE: bob.md---
# Bob
Content about Bob
---END_FILE---
===END_PEOPLE===

===TOPICS===
===END_TOPICS==="""
        _, people, _ = _parse_writer_output(text)
        assert len(people) == 2
        assert "alice.md" in people
        assert "bob.md" in people


# ---------------------------------------------------------------------------
# Session filename helpers
# ---------------------------------------------------------------------------


class TestSessionFilename:
    def test_time_slot_morning(self) -> None:
        assert _time_slot(datetime(2026, 1, 1, 8, 0, tzinfo=UTC)) == "morning"

    def test_time_slot_afternoon(self) -> None:
        assert _time_slot(datetime(2026, 1, 1, 14, 0, tzinfo=UTC)) == "afternoon"

    def test_time_slot_evening(self) -> None:
        assert _time_slot(datetime(2026, 1, 1, 20, 0, tzinfo=UTC)) == "evening"

    def test_time_slot_night(self) -> None:
        assert _time_slot(datetime(2026, 1, 1, 3, 0, tzinfo=UTC)) == "night"

    def test_session_filename_for_slot(self) -> None:
        """Pure function of date+slot — no suffix, no store lookup."""
        dt = datetime(2026, 4, 13, 20, 0, tzinfo=UTC)
        assert _session_filename(dt) == "sessions/2026-04-13-evening.md"


# ---------------------------------------------------------------------------
# MemoryWriter.run() tests
# ---------------------------------------------------------------------------


class TestMemoryWriterRun:
    def test_no_unsummarized_turns_returns_empty(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        llm = FakeLLMClient()
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert result.turns_summarized == 0
        assert result.session_file is None
        assert llm.calls == []  # No LLM call made

    def test_writes_session_file(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert result.session_file is not None
        assert result.session_file.startswith("sessions/")
        content = mem_store.read_file(result.session_file)
        assert "Alice and the familiar" in content

    def test_creates_people_file(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert "people/discord-1.md" in result.people_files
        content = mem_store.read_file("people/discord-1.md")
        assert "# Alice" in content

    def test_creates_topic_file(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert "topics/greetings.md" in result.topic_files
        content = mem_store.read_file("topics/greetings.md")
        assert "# Greetings" in content

    def test_advances_watermark(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert result.watermark_id > 0
        wm = hist_store.get_writer_watermark(familiar_id=_FAMILIAR)
        assert wm is not None
        assert wm.last_written_id == result.watermark_id

    def test_no_watermark_advance_on_llm_failure(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)

        class FailingLLM(FakeLLMClient):
            async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
                msg = "LLM failed"
                raise RuntimeError(msg)

        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=FailingLLM(),
            familiar_id=_FAMILIAR,
        )
        with pytest.raises(RuntimeError, match="LLM failed"):
            asyncio.run(writer.run())

        # Watermark should NOT have advanced
        assert hist_store.get_writer_watermark(familiar_id=_FAMILIAR) is None

    def test_empty_llm_response_returns_empty_result(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[""])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert result.session_file is None
        assert result.turns_summarized == 0

    def test_existing_people_file_included_in_prompt(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        # Create an existing people file for Alice keyed by canonical slug
        mem_store.write_file(
            "people/discord-1.md",
            "# Alice\nExisting info about Alice.",
            source="test",
        )
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())

        # Check the prompt included the existing file content
        assert len(llm.calls) == 1
        user_msg = llm.calls[0][1].content
        assert "Existing info about Alice" in user_msg

    def test_audit_source_tag(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 10)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())
        for entry in mem_store.audit_entries:
            assert entry.source == "memory_writer"

    def test_turns_summarized_count(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 6)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())
        assert result.turns_summarized == 6

    def test_second_pass_updates_same_session_file(self, tmp_path: Path) -> None:
        """Two writer runs in the same slot update one file, not two.

        Regression for the ``-2``/``-3`` suffix bug: the writer used
        to treat every invocation as a fresh session, fragmenting a
        single conversation across multiple files.
        """
        mem_store, hist_store = _make_stores(tmp_path)

        # first pass: 10 turns, produces sessions/<date>-<slot>.md
        _seed_turns(hist_store, 10)
        llm1 = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm1,
            familiar_id=_FAMILIAR,
        )
        first = asyncio.run(writer.run())
        assert first.session_file is not None

        # second pass in the same slot: more turns, same writer
        _seed_turns(hist_store, 10)
        second_output = _STRUCTURED_OUTPUT.replace(
            "Alice and the familiar had a brief exchange about greetings.",
            "Alice and the familiar continued chatting; merged summary here.",
        )
        llm2 = FakeLLMClient(replies=[second_output])
        writer2 = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm2,
            familiar_id=_FAMILIAR,
        )
        second = asyncio.run(writer2.run())

        assert second.session_file == first.session_file, (
            "second pass must reuse the same session file instead of "
            "creating a -2 suffix"
        )
        # and only one session file should exist on disk
        assert mem_store.glob("sessions/*.md") == [first.session_file]
        content = mem_store.read_file(first.session_file)
        assert "merged summary here" in content

    def test_existing_session_content_included_in_prompt(self, tmp_path: Path) -> None:
        """Prior session summary is fed back so the LLM can merge, not restate."""
        mem_store, hist_store = _make_stores(tmp_path)

        # first pass writes an initial summary
        _seed_turns(hist_store, 6)
        llm1 = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm1,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())

        # second pass: prompt must carry the prior session text
        _seed_turns(hist_store, 6)
        llm2 = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer2 = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm2,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer2.run())

        assert len(llm2.calls) == 1
        user_msg = llm2.calls[0][1].content
        assert "Alice and the familiar had a brief exchange about greetings" in user_msg

    def test_filenames_with_directory_prefix_are_normalized(
        self, tmp_path: Path
    ) -> None:
        """LLM emitting ``people/<slug>.md`` must not produce nested directories.

        Regression test: the existing-files section shows paths like
        ``people/<slug>.md``, so the model can mirror that in its output.
        """
        prefixed_output = """\
===SESSION_SUMMARY===
Alice and the familiar had a brief exchange.
===END_SESSION_SUMMARY===

===PEOPLE===
---FILE: people/discord-1.md---
# Alice

Alice is a friendly person.
---END_FILE---
===END_PEOPLE===

===TOPICS===
---FILE: topics/greetings.md---
# Greetings

A casual chat.
---END_FILE---
===END_TOPICS==="""
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)
        llm = FakeLLMClient(replies=[prefixed_output])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        result = asyncio.run(writer.run())

        assert "people/discord-1.md" in result.people_files
        assert "people/people/discord-1.md" not in result.people_files
        assert "topics/greetings.md" in result.topic_files
        assert "topics/topics/greetings.md" not in result.topic_files
        # and the files on disk live at the flat location
        assert mem_store.read_file("people/discord-1.md").startswith("# Alice")
        assert mem_store.read_file("topics/greetings.md").startswith("# Greetings")


# ---------------------------------------------------------------------------
# Channel context header — thread/forum labels flow into the writer prompt
# ---------------------------------------------------------------------------


class TestChannelContextBlock:
    def test_no_lookup_omits_context_block(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())
        prompt = llm.calls[0][1].content
        assert "## Context" not in prompt

    def test_lookup_prepends_context_block(self, tmp_path: Path) -> None:
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)  # seeded at channel_id=100
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
            channel_context_lookup=lambda cid: (
                "#general -> brainstorm" if cid == 100 else str(cid)
            ),
        )
        asyncio.run(writer.run())
        prompt = llm.calls[0][1].content
        assert "## Context" in prompt
        assert "- #general -> brainstorm" in prompt
        # block sits at the top of the transcript body (above first turn)
        assert prompt.index("## Context") < prompt.index("user (Alice): turn 0")

    def test_lookup_unknown_channel_skips_block(self, tmp_path: Path) -> None:
        """Lookup returning bare id for every channel yields no block."""
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
            channel_context_lookup=str,
        )
        asyncio.run(writer.run())
        prompt = llm.calls[0][1].content
        assert "## Context" not in prompt

    def test_multi_channel_context_block(self, tmp_path: Path) -> None:
        """Turns across two channels produce two context lines in order."""
        mem_store, hist_store = _make_stores(tmp_path)
        # two channels, interleaved
        for i in range(4):
            cid = 100 if i % 2 == 0 else 200
            hist_store.append_turn(
                familiar_id=_FAMILIAR,
                channel_id=cid,
                role="user",
                content=f"msg {i}",
                author=_ALICE,
            )
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        labels = {100: "#general", 200: "#general -> thread"}
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
            channel_context_lookup=lambda cid: labels.get(cid, str(cid)),
        )
        asyncio.run(writer.run())
        prompt = llm.calls[0][1].content
        assert "- #general\n- #general -> thread" in prompt


# ---------------------------------------------------------------------------
# Author roster + alias index — canonical slug as filename
# ---------------------------------------------------------------------------


class TestAuthorRoster:
    def test_roster_includes_canonical_slug_and_known_names(
        self, tmp_path: Path
    ) -> None:
        """Prompt roster lists each transcript author with their canonical slug."""
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())
        prompt = llm.calls[0][1].content
        assert "canonical slug: discord-1" in prompt
        assert "Alice" in prompt

    def test_roster_absent_when_no_user_turns(self, tmp_path: Path) -> None:
        """Assistant-only transcripts emit no author roster block."""
        mem_store, hist_store = _make_stores(tmp_path)
        for i in range(4):
            hist_store.append_turn(
                familiar_id=_FAMILIAR,
                channel_id=100,
                role="assistant",
                content=f"t{i}",
            )
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())
        prompt = llm.calls[0][1].content
        assert "canonical slug" not in prompt


class TestAliasIndex:
    def test_alias_index_written_after_people_pass(self, tmp_path: Path) -> None:
        """``people/_aliases.json`` maps every known name to the canonical slug."""
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())

        raw = mem_store.read_file("people/_aliases.json")
        index = json.loads(raw)
        assert index["alice"] == "discord-1"

    def test_alias_section_in_file_feeds_index(self, tmp_path: Path) -> None:
        """Any ``## Aliases`` bullet list in a people file is picked up."""
        mem_store, hist_store = _make_stores(tmp_path)
        _seed_turns(hist_store, 4)
        llm = FakeLLMClient(replies=[_STRUCTURED_OUTPUT])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())

        # Alice's file (from _STRUCTURED_OUTPUT) includes ## Aliases: Alice, alice.
        raw = mem_store.read_file("people/_aliases.json")
        index = json.loads(raw)
        assert index.get("alice") == "discord-1"

    def test_no_index_written_when_no_authors_or_files(self, tmp_path: Path) -> None:
        """Assistant-only run with no people writes skips the index."""
        mem_store, hist_store = _make_stores(tmp_path)
        for i in range(4):
            hist_store.append_turn(
                familiar_id=_FAMILIAR,
                channel_id=100,
                role="assistant",
                content=f"t{i}",
            )
        # Output with no people files
        empty_people_output = """\
===SESSION_SUMMARY===
Quiet.
===END_SESSION_SUMMARY===

===PEOPLE===
===END_PEOPLE===

===TOPICS===
===END_TOPICS==="""
        llm = FakeLLMClient(replies=[empty_people_output])
        writer = MemoryWriter(
            memory_store=mem_store,
            history_store=hist_store,
            llm_client=llm,
            familiar_id=_FAMILIAR,
        )
        asyncio.run(writer.run())
        assert mem_store.glob("people/_aliases.json") == []
