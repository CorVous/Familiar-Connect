"""SillyTavern lorebook / world-info importer.

Step 9 of docs/architecture/context-pipeline.md. Reads a SillyTavern
lorebook JSON and writes one Markdown file per entry into a
subdirectory of a familiar's :class:`MemoryStore` (default
``lore/imported/``). Each output file is plain Markdown with the
entry's comment as the H1, the trigger keywords as a blockquoted
bulleted list at the top (kept for human reference; the runtime
never reads them at search time), and the entry body as the
content.

Once imported, the files are indistinguishable from any other
Markdown in the memory directory — the agentic
:class:`ContentSearchProvider` finds them via grep just like
anything else. There is no runtime keyword walker, no World Info
trigger logic, and no special-cased data path. Imports are a
one-shot translation, not an ongoing dependency.

The importer is non-fatal at the entry level: malformed entries,
oversized entries, and entries that would overwrite an existing
file (without ``force=True``) are recorded in
:class:`ImportResult`'s ``errors`` / ``skipped`` lists rather than
aborting the whole import. Top-level errors (unreadable file,
invalid JSON, missing ``entries`` field) raise
:class:`LorebookImportError`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from familiar_connect.memory.store import MemoryStoreError, MemoryStorePathError

if TYPE_CHECKING:
    from familiar_connect.memory.store import MemoryStore


PathLike = str | Path

_AUDIT_SOURCE = "silly_tavern_lorebook_importer"
_DEFAULT_TARGET_DIR = "lore/imported"
_MAX_SLUG_LENGTH = 50


class LorebookImportError(Exception):
    """Top-level failure during import — bad source, bad JSON, bad shape."""


@dataclass
class ImportResult:
    """Per-import outcome record.

    :param written: Relative paths of files actually created.
    :param skipped: Relative paths that already existed and were left
        untouched (because ``force=False``).
    :param errors: Human-readable per-entry error messages. Each entry
        that couldn't be imported produces exactly one string here.
    """

    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def import_silly_tavern_lorebook(
    store: MemoryStore,
    source: dict[str, Any] | bytes | str | Path,
    *,
    target_dir: str = _DEFAULT_TARGET_DIR,
    force: bool = False,
) -> ImportResult:
    """Import a SillyTavern lorebook into *store*.

    :param store: The familiar's :class:`MemoryStore`.
    :param source: One of:

        - an already-parsed lorebook ``dict``,
        - raw JSON ``bytes`` or ``str``,
        - a :class:`Path` (or path string) pointing to a JSON file on
          disk.
    :param target_dir: Relative subdirectory under the store root to
        write the imported files into. Defaults to ``lore/imported``.
        Must not escape the store root — path-traversal attempts
        propagate as :class:`MemoryStorePathError`.
    :param force: If ``True``, existing files at the destination paths
        are overwritten. If ``False`` (the default), they are left
        untouched and recorded in :attr:`ImportResult.skipped`.
    :return: A populated :class:`ImportResult`.
    :raises LorebookImportError: If the source can't be loaded, the
        JSON can't be parsed, or the document doesn't have a
        well-formed ``entries`` field.
    """
    book = _load_source(source)
    entries = _coerce_entries(book)

    result = ImportResult()
    used_slugs: set[str] = set()

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            result.errors.append(
                f"entry is not a dict (got {type(raw_entry).__name__})"
            )
            continue

        if raw_entry.get("disable") is True:
            continue

        try:
            label = _entry_label(raw_entry)
            content = _entry_content(raw_entry)
        except _EntryShapeError as exc:
            result.errors.append(f"{exc.label}: {exc}")
            continue

        slug = _unique_slug(_slugify(label) or "entry", used_slugs)
        used_slugs.add(slug)

        rel_path = f"{target_dir.rstrip('/')}/{slug}.md"
        markdown = _render_markdown(label=label, content=content, entry=raw_entry)

        # _file_exists() lets MemoryStorePathError propagate naturally;
        # a malformed target_dir is a programmer error, not a per-entry
        # recoverable failure.
        if not force and _file_exists(store, rel_path):
            result.skipped.append(rel_path)
            continue

        try:
            store.write_file(rel_path, markdown, source=_AUDIT_SOURCE)
        except MemoryStoreError as exc:
            # Path-traversal is the one store error that escapes the
            # per-entry recoverable bucket — re-raise so the caller sees
            # the real reason their target_dir was rejected.
            if isinstance(exc, MemoryStorePathError):
                raise
            result.errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue

        result.written.append(rel_path)

    return result


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _load_source(
    source: dict[str, Any] | bytes | str | Path,
) -> dict[str, Any]:
    if isinstance(source, dict):
        return source

    if isinstance(source, Path):
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"cannot read lorebook file {source}: {exc}"
            raise LorebookImportError(msg) from exc
        return _parse_json_text(text)

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = f"lorebook bytes are not valid UTF-8: {exc}"
            raise LorebookImportError(msg) from exc
        return _parse_json_text(text)

    if isinstance(source, str):
        # Heuristic: a string starting with whitespace then '{' or '[' is
        # raw JSON; anything else is treated as a path.
        stripped = source.lstrip()
        if stripped.startswith(("{", "[")):
            return _parse_json_text(source)
        return _load_source(Path(source))

    msg = f"unsupported source type: {type(source).__name__}"
    raise LorebookImportError(msg)


def _parse_json_text(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"lorebook JSON is invalid: {exc}"
        raise LorebookImportError(msg) from exc
    if not isinstance(obj, dict):
        msg = f"lorebook root must be an object, got {type(obj).__name__}"
        raise LorebookImportError(msg)
    return obj


def _coerce_entries(book: dict[str, Any]) -> list[Any]:
    """Return a list of raw entry values, *unvalidated*.

    Per-entry shape validation happens later in the main loop, so a
    single bad entry can be recorded as an error without aborting
    the whole import.
    """
    if "entries" not in book:
        msg = "lorebook is missing the 'entries' field"
        raise LorebookImportError(msg)
    raw = book["entries"]
    if isinstance(raw, dict):
        # SillyTavern's default export uses string keys "0", "1", … as
        # the dict shape; the values are the actual entries.
        return list(raw.values())
    if isinstance(raw, list):
        return list(raw)
    msg = f"'entries' must be a dict or list, got {type(raw).__name__}"
    raise LorebookImportError(msg)


# ---------------------------------------------------------------------------
# Per-entry shape extraction
# ---------------------------------------------------------------------------


class _EntryShapeError(ValueError):
    """An entry was missing required fields. Caught and recorded."""

    def __init__(self, label: str, message: str) -> None:
        super().__init__(message)
        self.label = label


def _entry_label(entry: dict[str, Any]) -> str:
    """Return a human-readable label for *entry* (used as the H1)."""
    comment = entry.get("comment")
    if isinstance(comment, str) and comment.strip():
        return comment.strip()

    keys = entry.get("key")
    if isinstance(keys, list) and keys:
        first = keys[0]
        if isinstance(first, str) and first.strip():
            return first.strip()

    uid = entry.get("uid")
    if isinstance(uid, int):
        return f"uid {uid}"

    return "entry"


def _entry_content(entry: dict[str, Any]) -> str:
    content = entry.get("content")
    if not isinstance(content, str):
        raise _EntryShapeError(
            _entry_label(entry),
            f"content must be a string, got {type(content).__name__}",
        )
    return content


# ---------------------------------------------------------------------------
# Slugging
# ---------------------------------------------------------------------------


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Lowercase, strip non-ASCII-alnum, collapse to single hyphens."""
    lowered = text.lower()
    # Replace any run of non-alnum with a single hyphen.
    slug = _SLUG_NON_ALNUM.sub("-", lowered).strip("-")
    return slug[:_MAX_SLUG_LENGTH]


def _unique_slug(base: str, used: set[str]) -> str:
    """Return *base*, or ``base-2``/``base-3``/… if it's already taken."""
    if base not in used:
        return base
    n = 2
    while True:
        candidate = f"{base}-{n}"
        if candidate not in used:
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(
    *,
    label: str,
    content: str,
    entry: dict[str, Any],
) -> str:
    sections: list[str] = [f"# {label}"]

    keys = entry.get("key")
    if isinstance(keys, list):
        clean_keys = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
        if clean_keys:
            block = (
                "> Trigger keywords (preserved for reference, not used at runtime):\n"
                + "\n".join(f"> - {k}" for k in clean_keys)
            )
            sections.append(block)

    sections.append(content.strip())
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Store probing — exists check via the public API
# ---------------------------------------------------------------------------


def _file_exists(store: MemoryStore, rel_path: str) -> bool:
    """Return True if a file at *rel_path* already exists in the store.

    Uses :meth:`MemoryStore.read_file` as the exists-check so the
    path-traversal validation in the store's resolver still applies.
    A "not found" error returns False; any other store error
    propagates.
    """
    try:
        store.read_file(rel_path)
    except MemoryStorePathError:
        # Path-traversal: re-raise so the caller surfaces the real reason.
        raise
    except MemoryStoreError:
        return False
    return True
