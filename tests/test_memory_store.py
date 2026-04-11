"""Red-first tests for the per-familiar MemoryStore.

Covers familiar_connect.memory.store, which does not exist yet.

The store owns a single per-familiar directory of plain-text files
(see docs/architecture/memory.md). It
exposes a small file-IO surface — list / read / write / append / grep
/ glob — that is safe to hand to a tool-using cheap model later via
the ContentSearchProvider:

  - Every operation is scoped to the root and rejects path-traversal
    (`..`, absolute paths, symlinks pointing outside).
  - Per-file size, per-operation result count, and per-directory file
    count are capped via a configurable MemoryStoreLimits dataclass.
  - Every successful write goes through an in-memory audit log so the
    pipeline can reconstruct "when did the bot's beliefs about Alice
    change."
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from familiar_connect.memory.store import (
    AuditEntry,
    GrepHit,
    MemoryEntry,
    MemoryStore,
    MemoryStoreError,
    MemoryStoreLimits,
    MemoryStorePathError,
    MemoryStoreSizeLimitError,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Return a fresh MemoryStore rooted at a pytest-managed temp directory."""
    return MemoryStore(tmp_path / "memory")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_root_if_missing(self, tmp_path: Path) -> None:
        root = tmp_path / "memory"
        assert not root.exists()
        MemoryStore(root)
        assert root.is_dir()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        root = tmp_path / "memory"
        MemoryStore(str(root))
        assert root.is_dir()

    def test_accepts_existing_root(self, tmp_path: Path) -> None:
        root = tmp_path / "memory"
        root.mkdir()
        (root / "preexisting.md").write_text("already here", encoding="utf-8")
        s = MemoryStore(root)
        assert s.read_file("preexisting.md") == "already here"

    def test_uses_default_limits(self, store: MemoryStore) -> None:
        # The default per-file cap is 256 KB.
        assert store.limits.max_file_bytes == 256 * 1024
        assert store.limits.max_results_per_op > 0
        assert store.limits.max_files_per_dir > 0

    def test_accepts_custom_limits(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(
            max_file_bytes=128,
            max_results_per_op=5,
            max_files_per_dir=3,
        )
        s = MemoryStore(tmp_path / "memory", limits=limits)
        assert s.limits.max_file_bytes == 128
        assert s.limits.max_results_per_op == 5
        assert s.limits.max_files_per_dir == 3


# ---------------------------------------------------------------------------
# write_file / read_file
# ---------------------------------------------------------------------------


class TestWriteAndRead:
    def test_round_trip(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "hello world")
        assert store.read_file("notes.md") == "hello world"

    def test_creates_intermediate_directories(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "she likes cats")
        assert store.read_file("people/alice.md") == "she likes cats"

    def test_overwrites_existing_file(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "first")
        store.write_file("notes.md", "second")
        assert store.read_file("notes.md") == "second"

    def test_unicode_round_trip(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "Aria says — “hello” 🦊")
        assert store.read_file("notes.md") == "Aria says — “hello” 🦊"

    def test_atomic_write_no_temp_file_left_behind(
        self, store: MemoryStore, tmp_path: Path
    ) -> None:
        store.write_file("notes.md", "hello")
        leftovers = list((tmp_path / "memory").glob("**/*.tmp*"))
        assert leftovers == []

    def test_read_missing_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.read_file("nonexistent.md")

    def test_read_directory_raises(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        with pytest.raises(MemoryStoreError):
            store.read_file("people")


# ---------------------------------------------------------------------------
# append_file
# ---------------------------------------------------------------------------


class TestAppend:
    def test_append_to_existing(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "hello")
        store.append_file("notes.md", " world")
        assert store.read_file("notes.md") == "hello world"

    def test_append_to_missing_creates(self, store: MemoryStore) -> None:
        store.append_file("new.md", "first line\n")
        assert store.read_file("new.md") == "first line\n"

    def test_append_creates_intermediate_dirs(self, store: MemoryStore) -> None:
        store.append_file("topics/elden-ring.md", "great game")
        assert store.read_file("topics/elden-ring.md") == "great game"


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


class TestListDir:
    def test_empty_root(self, store: MemoryStore) -> None:
        assert store.list_dir() == []

    def test_lists_top_level_files_and_dirs(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "x")
        store.write_file("people/alice.md", "y")
        entries = {e.name: e for e in store.list_dir()}
        assert set(entries) == {"notes.md", "people"}
        assert entries["notes.md"].is_dir is False
        assert entries["people"].is_dir is True

    def test_lists_subdirectory(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        store.write_file("people/bob.md", "y")
        names = sorted(e.name for e in store.list_dir("people"))
        assert names == ["alice.md", "bob.md"]

    def test_returns_memory_entry_with_size(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "12345")
        (entry,) = store.list_dir()
        assert isinstance(entry, MemoryEntry)
        assert entry.size_bytes == 5
        assert isinstance(entry.modified, datetime)

    def test_directory_entry_has_zero_size(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        entries = {e.name: e for e in store.list_dir()}
        assert entries["people"].size_bytes == 0

    def test_missing_dir_raises(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.list_dir("does/not/exist")


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class TestGlob:
    def test_top_level_glob(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "x")
        store.write_file("readme.txt", "y")
        store.write_file("people/alice.md", "z")
        results = store.glob("*.md")
        assert results == ["notes.md"]

    def test_recursive_glob(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "x")
        store.write_file("people/alice.md", "y")
        store.write_file("topics/games/elden-ring.md", "z")
        results = sorted(store.glob("**/*.md"))
        assert results == [
            "notes.md",
            "people/alice.md",
            "topics/games/elden-ring.md",
        ]

    def test_glob_returns_relative_paths(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "x")
        results = store.glob("**/*.md")
        # No absolute paths, no leading slash, no leading "memory/".
        for r in results:
            assert not r.startswith("/")
            assert "memory" not in r.split("/")[0]


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    def test_finds_match_in_single_file(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "she likes cats and dogs")
        hits = store.grep("cats")
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, GrepHit)
        assert hit.rel_path == "people/alice.md"
        assert hit.line_number == 1
        assert "cats" in hit.line_text

    def test_finds_matches_across_multiple_files(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "alice likes cats")
        store.write_file("people/bob.md", "bob also likes cats")
        store.write_file("notes.md", "no match here")
        hits = store.grep("cats")
        rels = sorted(h.rel_path for h in hits)
        assert rels == ["people/alice.md", "people/bob.md"]

    def test_case_insensitive_default(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "She Loves Cats")
        hits = store.grep("cats")
        assert len(hits) == 1

    def test_case_sensitive_when_flag_off(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "She Loves Cats")
        hits = store.grep("cats", case_insensitive=False)
        assert hits == []

    def test_scoped_to_subdirectory(self, store: MemoryStore) -> None:
        store.write_file("people/alice.md", "alice likes cats")
        store.write_file("topics/cats.md", "topic about cats")
        hits = store.grep("cats", rel_path="people")
        rels = [h.rel_path for h in hits]
        assert rels == ["people/alice.md"]

    def test_line_numbers_are_one_based(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "first line\nsecond cats line\nthird line")
        hits = store.grep("cats")
        assert len(hits) == 1
        assert hits[0].line_number == 2

    def test_empty_pattern_returns_no_hits(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "anything")
        assert store.grep("") == []

    def test_regex_pattern(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "user42 logged in\nadmin99 logged in")
        hits = store.grep(r"user\d+")
        assert len(hits) == 1
        assert "user42" in hits[0].line_text

    def test_skips_files_over_size_limit(self, tmp_path: Path) -> None:
        """Files larger than max_file_bytes are skipped, not crashed on."""
        limits = MemoryStoreLimits(max_file_bytes=64)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        # Write two files directly to the disk so we bypass the write-side
        # size check and exercise the read-side defensive skip.
        small = tmp_path / "memory" / "small.md"
        big = tmp_path / "memory" / "big.md"
        small.write_text("cats", encoding="utf-8")
        big.write_text("cats" * 100, encoding="utf-8")  # 400 bytes

        hits = s.grep("cats")
        rels = [h.rel_path for h in hits]
        assert rels == ["small.md"]


# ---------------------------------------------------------------------------
# Path-traversal safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    @pytest.mark.parametrize(
        "bad_path",
        [
            "../escape.md",
            "../../escape.md",
            "people/../../escape.md",
            "people/../../../etc/passwd",
        ],
    )
    def test_relative_dot_dot_rejected(self, store: MemoryStore, bad_path: str) -> None:
        with pytest.raises(MemoryStorePathError):
            store.write_file(bad_path, "x")

    def test_absolute_path_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStorePathError):
            store.write_file("/tmp/escape.md", "x")  # noqa: S108

    def test_null_byte_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStorePathError):
            store.write_file("notes\x00.md", "x")

    def test_read_traversal_rejected(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "x")
        with pytest.raises(MemoryStorePathError):
            store.read_file("../escape.md")

    def test_list_traversal_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStorePathError):
            store.list_dir("../")

    def test_grep_traversal_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStorePathError):
            store.grep("anything", rel_path="../")

    def test_glob_with_absolute_pattern_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStorePathError):
            store.glob("/etc/*")

    def test_symlink_pointing_outside_rejected(
        self,
        store: MemoryStore,
        tmp_path: Path,
    ) -> None:
        """A symlink that escapes the root must not be followable."""
        outside = tmp_path / "outside.md"
        outside.write_text("secret", encoding="utf-8")
        link = tmp_path / "memory" / "link.md"
        link.symlink_to(outside)

        with pytest.raises(MemoryStorePathError):
            store.read_file("link.md")


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------


class TestSizeLimits:
    def test_write_over_max_file_bytes_rejected(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_file_bytes=16)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        with pytest.raises(MemoryStoreSizeLimitError):
            s.write_file("big.md", "x" * 17)

    def test_write_at_boundary_succeeds(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_file_bytes=16)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        s.write_file("ok.md", "x" * 16)
        assert s.read_file("ok.md") == "x" * 16

    def test_failed_write_does_not_create_file(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_file_bytes=4)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        with pytest.raises(MemoryStoreSizeLimitError):
            s.write_file("big.md", "too long")
        assert not (tmp_path / "memory" / "big.md").exists()

    def test_append_pushing_over_limit_rejected(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_file_bytes=10)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        s.write_file("notes.md", "12345")
        with pytest.raises(MemoryStoreSizeLimitError):
            s.append_file("notes.md", "678901")  # would total 11
        # Original content is preserved.
        assert s.read_file("notes.md") == "12345"

    def test_list_dir_over_max_files_per_dir_raises(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_files_per_dir=3)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        for i in range(5):
            s.write_file(f"f{i}.md", "x")
        with pytest.raises(MemoryStoreSizeLimitError):
            s.list_dir()

    def test_grep_caps_results_at_max_results_per_op(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_results_per_op=2)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        for i in range(5):
            s.write_file(f"f{i}.md", "match")
        hits = s.grep("match")
        assert len(hits) == 2


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_starts_empty(self, store: MemoryStore) -> None:
        assert store.audit_entries == []

    def test_write_records_entry(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "hello", source="test")
        entries = store.audit_entries
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, AuditEntry)
        assert e.rel_path == "notes.md"
        assert e.operation == "write"
        assert e.bytes_written == 5
        assert e.source == "test"
        assert isinstance(e.timestamp, datetime)

    def test_append_records_entry(self, store: MemoryStore) -> None:
        store.write_file("notes.md", "hello", source="seed")
        store.append_file("notes.md", " world", source="user")
        entries = store.audit_entries
        assert [e.operation for e in entries] == ["write", "append"]
        assert entries[1].source == "user"
        assert entries[1].bytes_written == len(b" world")

    def test_failed_write_does_not_log(self, tmp_path: Path) -> None:
        limits = MemoryStoreLimits(max_file_bytes=4)
        s = MemoryStore(tmp_path / "memory", limits=limits)
        with pytest.raises(MemoryStoreSizeLimitError):
            s.write_file("big.md", "too long")
        assert s.audit_entries == []

    def test_default_source_is_unknown_or_similar(self, store: MemoryStore) -> None:
        """Calls without an explicit source still produce a non-empty marker."""
        store.write_file("notes.md", "x")
        assert store.audit_entries[0].source

    def test_audit_entries_are_a_copy(self, store: MemoryStore) -> None:
        """Mutating the returned list does not affect the store's internal log."""
        store.write_file("notes.md", "x")
        snapshot = store.audit_entries
        snapshot.clear()
        assert len(store.audit_entries) == 1
