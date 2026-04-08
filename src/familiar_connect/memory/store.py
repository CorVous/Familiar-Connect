"""On-disk MemoryStore for a single familiar.

Owns one directory of plain-text files (one tree per familiar per
guild). Exposes a small, deliberately boring file-IO surface — list /
read / write / append / glob / grep — that is safe to hand to a tool-
using cheap model later via the ContentSearchProvider.

Key invariants enforced here, not by every caller:

- **Path-traversal safety.** Every relative path is resolved against
  the store's root and rejected if the resolved real path is not under
  the root. Absolute paths, ``..`` segments that escape, null bytes,
  and symlinks pointing outside the store all raise
  :class:`MemoryStorePathError`.
- **Sanity caps.** Per-file size, per-operation result count, and per-
  directory file count are bounded by a configurable
  :class:`MemoryStoreLimits`. Hitting a cap raises
  :class:`MemoryStoreSizeLimitError` rather than reading or writing
  silently truncated content.
- **Atomic writes.** ``write_file`` and ``append_file`` write to a
  sibling temp file and ``rename`` it into place so a partial write
  is never observable.
- **Audit log.** Every successful write or append is recorded in an
  in-memory log so the pipeline can later reconstruct "when did the
  bot's beliefs about Alice change." The log is intentionally
  in-process for the first cut; persisting it is a follow-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PathLike = str | Path

DEFAULT_MAX_FILE_BYTES = 256 * 1024
"""Per-file byte cap. ~256 KB is plenty for a Markdown note about a
person, topic, session, or piece of lore, and keeps grep cheap."""

DEFAULT_MAX_RESULTS_PER_OP = 1000
"""Cap on the number of results a single read-side op may return.
Protects callers from accidentally pulling the entire store into
memory through a too-broad grep or glob."""

DEFAULT_MAX_FILES_PER_DIR = 10_000
"""Cap on entries returned by a single ``list_dir`` call. Same idea
as the result cap — defends against accidental enumerations of
runaway directories."""

_TMP_SUFFIX = ".__memstore_tmp__"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MemoryStoreError(Exception):
    """Base exception for all MemoryStore failures."""


class MemoryStorePathError(MemoryStoreError):
    """A path argument escaped the store's root or was otherwise unsafe."""


class MemoryStoreSizeLimitError(MemoryStoreError):
    """A per-file, per-op, or per-directory size cap was exceeded."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStoreLimits:
    """Configurable safety caps for a :class:`MemoryStore`."""

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_results_per_op: int = DEFAULT_MAX_RESULTS_PER_OP
    max_files_per_dir: int = DEFAULT_MAX_FILES_PER_DIR


@dataclass(frozen=True)
class MemoryEntry:
    """One entry returned by :meth:`MemoryStore.list_dir`."""

    name: str
    is_dir: bool
    size_bytes: int
    modified: datetime


@dataclass(frozen=True)
class GrepHit:
    """One match returned by :meth:`MemoryStore.grep`."""

    rel_path: str
    line_number: int
    line_text: str
    match_start: int
    match_end: int


@dataclass(frozen=True)
class AuditEntry:
    """A single record in the in-memory audit log."""

    rel_path: str
    operation: str  # "write" | "append"
    bytes_written: int
    source: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class _MutableState:
    audit: list[AuditEntry] = field(default_factory=list)


class MemoryStore:
    """Per-familiar plain-text memory directory.

    Construct with the absolute root path you want this familiar's
    memory to live under. The directory is created on first use if it
    doesn't already exist.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        limits: MemoryStoreLimits | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits or MemoryStoreLimits()
        self._state = _MutableState()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def list_dir(self, rel_path: str = "") -> list[MemoryEntry]:
        """Return the entries in the directory at *rel_path*.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreError: If the path doesn't exist or is not
            a directory.
        :raises MemoryStoreSizeLimitError: If the directory contains
            more than ``limits.max_files_per_dir`` entries.
        """
        path = self._resolve(rel_path)
        if not path.exists():
            msg = f"Directory not found: {rel_path or '.'}"
            raise MemoryStoreError(msg)
        if not path.is_dir():
            msg = f"Not a directory: {rel_path}"
            raise MemoryStoreError(msg)

        children = sorted(path.iterdir(), key=lambda p: p.name)
        if len(children) > self.limits.max_files_per_dir:
            msg = (
                f"Directory {rel_path or '.'} has {len(children)} entries, "
                f"exceeds limit of {self.limits.max_files_per_dir}"
            )
            raise MemoryStoreSizeLimitError(msg)

        entries: list[MemoryEntry] = []
        for child in children:
            stat = child.stat()
            is_dir = child.is_dir()
            entries.append(
                MemoryEntry(
                    name=child.name,
                    is_dir=is_dir,
                    size_bytes=0 if is_dir else stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                )
            )
        return entries

    def read_file(self, rel_path: str) -> str:
        """Return the UTF-8 contents of the file at *rel_path*.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreError: If the path is missing or not a
            regular file.
        :raises MemoryStoreSizeLimitError: If the file is larger than
            ``limits.max_file_bytes``.
        """
        path = self._resolve(rel_path)
        if not path.exists():
            msg = f"File not found: {rel_path}"
            raise MemoryStoreError(msg)
        if not path.is_file():
            msg = f"Not a regular file: {rel_path}"
            raise MemoryStoreError(msg)
        size = path.stat().st_size
        if size > self.limits.max_file_bytes:
            msg = (
                f"File {rel_path} is {size} bytes, exceeds "
                f"limit of {self.limits.max_file_bytes}"
            )
            raise MemoryStoreSizeLimitError(msg)
        return path.read_text(encoding="utf-8")

    def glob(self, pattern: str) -> list[str]:
        """Return relative paths under the root matching *pattern*.

        Uses :meth:`pathlib.Path.glob`. Recursive globs use ``**``.
        Results are deduplicated, sorted, and capped at
        ``limits.max_results_per_op``. Any path that — after symlink
        resolution — escapes the root is silently dropped (we never
        leak escaping symlinks; we just don't return them).

        :raises MemoryStorePathError: If the pattern itself looks like
            a traversal attempt (absolute, contains a null byte, etc.).
        """
        if not pattern:
            return []
        if pattern.startswith("/") or "\x00" in pattern:
            msg = f"Unsafe glob pattern: {pattern!r}"
            raise MemoryStorePathError(msg)

        results: list[str] = []
        for match in self.root.glob(pattern):
            if not self._is_under_root(match):
                continue
            if match == self.root:
                continue
            rel = match.relative_to(self.root).as_posix()
            results.append(rel)
            if len(results) >= self.limits.max_results_per_op:
                break
        return sorted(set(results))

    def grep(
        self,
        pattern: str,
        rel_path: str = "",
        *,
        case_insensitive: bool = True,
    ) -> list[GrepHit]:
        """Return regex-match hits across files under *rel_path*.

        Walks the file tree under *rel_path* and runs *pattern*
        against each line of each readable file. Files larger than
        ``limits.max_file_bytes`` are silently skipped (defensive —
        someone may have written a giant file directly to disk).
        Files that aren't valid UTF-8 are also skipped.

        Results are capped at ``limits.max_results_per_op``.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        """
        if not pattern:
            return []

        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)

        root = self._resolve(rel_path)
        if not root.exists():
            return []

        targets: list[Path]
        if root.is_file():
            targets = [root]
        else:
            walked = list(root.rglob("*"))
            walked.sort(key=str)
            targets = walked

        hits: list[GrepHit] = []
        for f in targets:
            if not f.is_file():
                continue
            if not self._is_under_root(f):
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size > self.limits.max_file_bytes:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            rel = f.relative_to(self.root).as_posix()
            for line_num, line in enumerate(text.splitlines(), start=1):
                m = regex.search(line)
                if m is None:
                    continue
                hits.append(
                    GrepHit(
                        rel_path=rel,
                        line_number=line_num,
                        line_text=line,
                        match_start=m.start(),
                        match_end=m.end(),
                    )
                )
                if len(hits) >= self.limits.max_results_per_op:
                    return hits
        return hits

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def write_file(
        self,
        rel_path: str,
        content: str,
        *,
        source: str = "unknown",
    ) -> None:
        """Write *content* to the file at *rel_path*, replacing it.

        Atomic via temp-file + rename — a partial write is never
        observable. Creates intermediate directories. Records an
        ``AuditEntry`` on success.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreSizeLimitError: If *content* exceeds the
            per-file size cap.
        """
        path = self._resolve(rel_path)
        encoded = content.encode("utf-8")
        self._check_size(rel_path, len(encoded))
        self._atomic_write(path, encoded)
        self._record_audit(rel_path, "write", len(encoded), source)

    def append_file(
        self,
        rel_path: str,
        content: str,
        *,
        source: str = "unknown",
    ) -> None:
        """Append *content* to the file at *rel_path*, creating it if missing.

        Reads the existing file (if any), concatenates *content*, and
        writes the result atomically. The post-append size is checked
        against the cap; if it would overflow, the original file is
        left untouched and :class:`MemoryStoreSizeLimitError` is
        raised.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreSizeLimitError: If the post-append size
            exceeds the per-file size cap.
        """
        path = self._resolve(rel_path)

        existing = b""
        if path.exists():
            if not path.is_file():
                msg = f"Not a regular file: {rel_path}"
                raise MemoryStoreError(msg)
            existing = path.read_bytes()

        added = content.encode("utf-8")
        combined = existing + added
        self._check_size(rel_path, len(combined))
        self._atomic_write(path, combined)
        self._record_audit(rel_path, "append", len(added), source)

    # ------------------------------------------------------------------
    # Audit log accessor
    # ------------------------------------------------------------------

    @property
    def audit_entries(self) -> list[AuditEntry]:
        """Return a copy of the in-memory audit log."""
        return list(self._state.audit)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve *rel_path* against the root, rejecting any escape.

        Empty string maps to the root. Absolute paths, null bytes,
        and resolved paths outside the root all raise.
        """
        if not rel_path:
            return self.root
        if "\x00" in rel_path:
            msg = f"Path contains null byte: {rel_path!r}"
            raise MemoryStorePathError(msg)
        candidate = Path(rel_path)
        if candidate.is_absolute():
            msg = f"Absolute paths are not allowed: {rel_path}"
            raise MemoryStorePathError(msg)

        # Resolve through symlinks. We allow non-existent leaves so
        # write_file can target a file that doesn't yet exist; resolve
        # the parent that does exist plus the leaf, then check.
        full = (self.root / candidate).resolve(strict=False)
        if not self._is_under_root(full):
            msg = f"Path escapes the memory root: {rel_path}"
            raise MemoryStorePathError(msg)
        return full

    def _is_under_root(self, path: Path) -> bool:
        """Return True iff *path*, fully resolved, is under ``self.root``."""
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            return False
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return False
        return True

    def _check_size(self, rel_path: str, size_bytes: int) -> None:
        if size_bytes > self.limits.max_file_bytes:
            msg = (
                f"Write to {rel_path} would be {size_bytes} bytes, "
                f"exceeds limit of {self.limits.max_file_bytes}"
            )
            raise MemoryStoreSizeLimitError(msg)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + _TMP_SUFFIX)
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        finally:
            # If the rename happened, the temp file is gone; if it
            # didn't, we want to leave no garbage behind.
            if tmp.exists():
                tmp.unlink()

    def _record_audit(
        self,
        rel_path: str,
        operation: str,
        bytes_written: int,
        source: str,
    ) -> None:
        self._state.audit.append(
            AuditEntry(
                rel_path=rel_path,
                operation=operation,
                bytes_written=bytes_written,
                source=source,
                timestamp=datetime.now(tz=UTC),
            )
        )
