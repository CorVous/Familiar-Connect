"""Red-first tests for the SillyTavern lorebook / world-info importer.

Step 9 of future-features/context-management.md. Reads a SillyTavern
lorebook JSON file and writes one Markdown file per entry into a
subdirectory of the familiar's MemoryStore (default ``lore/imported``).
Each output file is plain Markdown — H1 from the entry's comment, the
trigger keywords as a blockquoted bulleted list at the top (kept for
human reference; the runtime never reads them), and the entry's
content as the body.

Covers familiar_connect.memory.import_silly_tavern, which doesn't
exist yet.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from familiar_connect.memory.import_silly_tavern import (
    ImportResult,
    LorebookImportError,
    import_silly_tavern_lorebook,
)
from familiar_connect.memory.store import MemoryStore, MemoryStoreError

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Return a fresh MemoryStore for each test."""
    return MemoryStore(tmp_path / "memory")


def _entry(
    *,
    uid: int = 0,
    comment: str = "",
    content: str = "x",
    key: list[str] | None = None,
    disable: bool = False,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "comment": comment,
        "content": content,
        "key": key if key is not None else [],
        "disable": disable,
    }


def _book(entries: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Wrap *entries* in a minimal SillyTavern-shaped lorebook dict.

    Loosely typed on purpose so error-path tests can feed
    intentionally malformed entries without fighting the type checker.
    """
    return {"name": "test lorebook", "entries": entries}


# ---------------------------------------------------------------------------
# Empty / smoke
# ---------------------------------------------------------------------------


class TestEmptyLorebook:
    def test_empty_dict_entries(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(store, _book({}))
        assert isinstance(result, ImportResult)
        assert result.written == []
        assert result.skipped == []
        assert result.errors == []

    def test_empty_list_entries(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(store, _book([]))
        assert result.written == []


# ---------------------------------------------------------------------------
# Single entry — happy path
# ---------------------------------------------------------------------------


class TestSingleEntry:
    def test_dict_entries_writes_one_file(self, store: MemoryStore) -> None:
        book = _book({
            "0": _entry(
                uid=0,
                comment="Alice character",
                content="Alice is a friendly cat with a fondness for tea.",
                key=["alice", "alice smith"],
            ),
        })
        result = import_silly_tavern_lorebook(store, book)

        assert len(result.written) == 1
        rel = result.written[0]
        assert rel.startswith("lore/imported/")
        assert rel.endswith(".md")

        text = store.read_file(rel)
        # H1 from the comment.
        assert "# Alice character" in text
        # Body from the content.
        assert "Alice is a friendly cat" in text

    def test_list_entries_writes_one_file(self, store: MemoryStore) -> None:
        book = _book([
            _entry(
                uid=0,
                comment="Alice character",
                content="A spirit who likes tea.",
                key=["alice"],
            ),
        ])
        result = import_silly_tavern_lorebook(store, book)
        assert len(result.written) == 1

    def test_keywords_block_present(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({
                "0": _entry(comment="Alice", content="x", key=["alice", "alice smith"])
            }),
        )
        text = store.read_file(result.written[0])
        assert "Trigger keywords" in text
        assert "alice" in text
        assert "alice smith" in text

    def test_no_keywords_no_keywords_block(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store, _book({"0": _entry(comment="Alice", content="x", key=[])})
        )
        text = store.read_file(result.written[0])
        assert "Trigger keywords" not in text


# ---------------------------------------------------------------------------
# Filename slugging
# ---------------------------------------------------------------------------


class TestSlugging:
    def test_slug_from_comment(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(comment="Alice Smith", content="x")}),
        )
        assert result.written == ["lore/imported/alice-smith.md"]

    def test_slug_punctuation_collapsed(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(comment="The Old Citadel!!  ", content="x")}),
        )
        assert result.written == ["lore/imported/the-old-citadel.md"]

    def test_slug_unicode_falls_back_to_ascii(self, store: MemoryStore) -> None:
        """Non-ASCII characters get stripped (or replaced)."""
        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(uid=7, comment="日本語タイトル", content="x")}),
        )
        # Either the slug stripped to nothing and we fell back to uid,
        # or it produced something ASCII. Either way it's a valid path.
        assert len(result.written) == 1
        rel = result.written[0]
        assert rel.startswith("lore/imported/")
        assert rel.endswith(".md")

    def test_slug_fallback_to_uid_when_comment_blank(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(uid=42, comment="", content="x")}),
        )
        assert "uid-42" in result.written[0]

    def test_slug_fallback_to_key_when_comment_and_uid_unusable(
        self, store: MemoryStore
    ) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(comment="", content="x", key=["castle of fire"])}),
        )
        assert "castle-of-fire" in result.written[0]

    def test_slug_collision_gets_numeric_suffix(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({
                "0": _entry(uid=0, comment="Alice", content="first"),
                "1": _entry(uid=1, comment="Alice", content="second"),
                "2": _entry(uid=2, comment="Alice", content="third"),
            }),
        )
        assert len(result.written) == 3
        # Three distinct paths, all alice-something.
        assert len(set(result.written)) == 3
        for rel in result.written:
            assert "alice" in rel


# ---------------------------------------------------------------------------
# Custom target directory
# ---------------------------------------------------------------------------


class TestTargetDir:
    def test_custom_target_dir(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(comment="Alice", content="x")}),
            target_dir="lore/from-st",
        )
        assert result.written == ["lore/from-st/alice.md"]


# ---------------------------------------------------------------------------
# force / skip semantics
# ---------------------------------------------------------------------------


class TestForceSemantics:
    def test_existing_file_skipped_without_force(self, store: MemoryStore) -> None:
        store.write_file("lore/imported/alice.md", "pre-existing")

        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(comment="Alice", content="from import")}),
        )
        assert result.written == []
        assert "lore/imported/alice.md" in result.skipped
        # The pre-existing file is preserved untouched.
        assert store.read_file("lore/imported/alice.md") == "pre-existing"

    def test_existing_file_overwritten_with_force(self, store: MemoryStore) -> None:
        store.write_file("lore/imported/alice.md", "pre-existing")

        result = import_silly_tavern_lorebook(
            store,
            _book({"0": _entry(comment="Alice", content="from import")}),
            force=True,
        )
        assert "lore/imported/alice.md" in result.written
        assert "from import" in store.read_file("lore/imported/alice.md")


# ---------------------------------------------------------------------------
# Source polymorphism — dict / bytes / str / Path
# ---------------------------------------------------------------------------


class TestSourcePolymorphism:
    def test_accepts_already_parsed_dict(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store, _book({"0": _entry(comment="Alice", content="x")})
        )
        assert len(result.written) == 1

    def test_accepts_json_string(self, store: MemoryStore) -> None:
        payload = json.dumps(_book({"0": _entry(comment="Alice", content="x")}))
        result = import_silly_tavern_lorebook(store, payload)
        assert len(result.written) == 1

    def test_accepts_json_bytes(self, store: MemoryStore) -> None:
        payload = json.dumps(_book({"0": _entry(comment="Alice", content="x")})).encode(
            "utf-8"
        )
        result = import_silly_tavern_lorebook(store, payload)
        assert len(result.written) == 1

    def test_accepts_file_path(self, store: MemoryStore, tmp_path: Path) -> None:
        path = tmp_path / "book.json"
        path.write_text(
            json.dumps(_book({"0": _entry(comment="Alice", content="x")})),
            encoding="utf-8",
        )
        result = import_silly_tavern_lorebook(store, path)
        assert len(result.written) == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_invalid_json_raises(self, store: MemoryStore) -> None:
        with pytest.raises(LorebookImportError, match="JSON"):
            import_silly_tavern_lorebook(store, "{not valid json")

    def test_missing_entries_field_raises(self, store: MemoryStore) -> None:
        with pytest.raises(LorebookImportError, match="entries"):
            import_silly_tavern_lorebook(store, {"name": "no entries"})

    def test_entries_wrong_type_raises(self, store: MemoryStore) -> None:
        with pytest.raises(LorebookImportError):
            import_silly_tavern_lorebook(store, {"entries": "not a list or dict"})

    def test_unreadable_file_raises(self, store: MemoryStore, tmp_path: Path) -> None:
        with pytest.raises(LorebookImportError):
            import_silly_tavern_lorebook(store, tmp_path / "nope.json")

    def test_individual_malformed_entry_recorded_in_errors(
        self, store: MemoryStore
    ) -> None:
        """One bad entry doesn't abort the whole import."""
        book = _book({
            "0": _entry(comment="Good", content="kept"),
            "1": "not even a dict",
            "2": _entry(comment="Also good", content="also kept"),
        })
        result = import_silly_tavern_lorebook(store, book)
        assert len(result.written) == 2
        assert len(result.errors) == 1

    def test_oversize_entry_recorded_in_errors(self, store: MemoryStore) -> None:
        """A single huge entry produces an error rather than aborting."""
        # Default max_file_bytes is 256 KB; produce a body that exceeds it.
        body = "x" * (300 * 1024)
        result = import_silly_tavern_lorebook(
            store,
            _book({
                "0": _entry(comment="Tiny", content="ok"),
                "1": _entry(comment="Huge", content=body),
            }),
        )
        assert len(result.written) == 1
        assert any("Huge" in err or "huge" in err for err in result.errors)

    def test_skips_disabled_entry(self, store: MemoryStore) -> None:
        """Entries with disable=True are skipped silently."""
        result = import_silly_tavern_lorebook(
            store,
            _book({
                "0": _entry(comment="Active", content="x"),
                "1": _entry(comment="Disabled", content="y", disable=True),
            }),
        )
        assert len(result.written) == 1
        # The disabled entry doesn't appear in errors either — it's a
        # deliberate omission.
        assert all("disabled" not in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# MemoryStore integration — writes go through the audit log
# ---------------------------------------------------------------------------


class TestMemoryStoreIntegration:
    def test_each_write_records_audit_entry(self, store: MemoryStore) -> None:
        result = import_silly_tavern_lorebook(
            store,
            _book({
                "0": _entry(comment="Alice", content="a"),
                "1": _entry(comment="Bob", content="b"),
            }),
        )
        sources = {e.source for e in store.audit_entries}
        assert all("silly_tavern" in s or "lorebook" in s for s in sources)
        assert len(store.audit_entries) == len(result.written)

    def test_import_then_grep_finds_content(self, store: MemoryStore) -> None:
        """Imported content is immediately searchable like any other file."""
        import_silly_tavern_lorebook(
            store,
            _book({
                "0": _entry(
                    comment="Alice",
                    content="She loves ska music and old citadels.",
                ),
            }),
        )
        hits = store.grep("ska")
        assert len(hits) == 1
        assert "lore/imported/alice.md" in hits[0].rel_path

    def test_path_traversal_in_target_dir_blocked(self, store: MemoryStore) -> None:
        """A malicious target_dir can't escape the store."""
        with pytest.raises(MemoryStoreError):
            import_silly_tavern_lorebook(
                store,
                _book({"0": _entry(comment="Alice", content="x")}),
                target_dir="../escape",
            )
