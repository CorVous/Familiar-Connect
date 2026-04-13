"""On-disk MemoryStore for a single familiar.

One plain-text directory tree per familiar per guild. Exposes
list / read / write / append / glob / grep for tool-using models.

Invariants enforced here, not by callers:

- Path-traversal safety — rejects escaping paths, symlinks, null bytes
- Sanity caps — per-file, per-op, per-dir limits via :class:`MemoryStoreLimits`
- Atomic writes — temp-file + rename; partial writes never observable
- Audit log — in-memory log of every write/append (persistence is a follow-up)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PathLike = str | Path

DEFAULT_MAX_FILE_BYTES = 256 * 1024
"""Per-file byte cap. ~256 KB covers any Markdown note and keeps grep cheap."""

DEFAULT_MAX_RESULTS_PER_OP = 1000
"""Cap on results per read-side op. Prevents pulling entire store
into memory via a too-broad grep or glob."""

DEFAULT_MAX_FILES_PER_DIR = 10_000
"""Cap on entries per ``list_dir`` call. Defends against runaway directories."""

_TMP_SUFFIX = ".__memstore_tmp__"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MemoryStoreError(Exception):
    """Base exception for all MemoryStore failures."""


class MemoryStorePathError(MemoryStoreError):
    """Path escaped store root or was otherwise unsafe."""


class MemoryStoreSizeLimitError(MemoryStoreError):
    """Size cap exceeded."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStoreLimits:
    """Safety caps for a :class:`MemoryStore`."""

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_results_per_op: int = DEFAULT_MAX_RESULTS_PER_OP
    max_files_per_dir: int = DEFAULT_MAX_FILES_PER_DIR


@dataclass(frozen=True)
class MemoryEntry:
    """Single entry from :meth:`MemoryStore.list_dir`."""

    name: str
    is_dir: bool
    size_bytes: int
    modified: datetime


@dataclass(frozen=True)
class GrepHit:
    """Single match from :meth:`MemoryStore.grep`."""

    rel_path: str
    line_number: int
    line_text: str
    match_start: int
    match_end: int


@dataclass(frozen=True)
class AuditEntry:
    """Single record in the in-memory audit log."""

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
    """Per-familiar plain-text memory directory."""

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
        """List entries in directory at *rel_path*.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreError: If path missing or not a directory.
        :raises MemoryStoreSizeLimitError: If entries exceed cap.
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
        """Return UTF-8 contents of file at *rel_path*.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreError: If path missing or not a regular file.
        :raises MemoryStoreSizeLimitError: If file exceeds byte cap.
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
        """Return matching relative paths, deduplicated and sorted.

        Capped at ``limits.max_results_per_op``. Escaping symlinks
        silently dropped.

        :raises MemoryStorePathError: If pattern is a traversal attempt.
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
        """Search files under *rel_path* for regex *pattern*.

        Oversized and non-UTF-8 files silently skipped.
        Capped at ``limits.max_results_per_op``.

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
        """Atomic replace. Creates intermediate directories.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreSizeLimitError: If *content* exceeds byte cap.
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
        """Append via atomic read-concat-write; create if missing.

        On overflow the original is left untouched.

        :raises MemoryStorePathError: If *rel_path* escapes the root.
        :raises MemoryStoreSizeLimitError: If post-append size exceeds cap.
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
        """Copy of the in-memory audit log."""
        return list(self._state.audit)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve *rel_path* against root; reject escapes.

        Empty string maps to root. Absolute paths, null bytes, and
        resolved paths outside root all raise.
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
        """Check whether *path*, fully resolved, is under ``self.root``."""
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
            # if the rename happened the temp file is gone; if it
            # didn't, leave no garbage behind
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
