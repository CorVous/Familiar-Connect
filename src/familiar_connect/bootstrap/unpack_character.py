"""Unpack a CharacterCard into a MemoryStore's ``self/`` directory.

See ``docs/architecture/context-pipeline.md``. Each non-empty card
field becomes a Markdown file under ``self/``; empty fields produce
no file.

Idempotent: re-unpacking identical card is a no-op. Differing
on-disk content raises :class:`CharacterUnpackError` unless
``overwrite=True``, in which case only changed fields are rewritten
and newly-empty fields are removed. All writes go through the
store's safety / atomic-write / audit machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.character import CharacterCard
    from familiar_connect.memory.store import MemoryStore


# mapping from CharacterCard attr to filename under ``self/``.
# order is canonical — callers and tests get a predictable list
_FIELD_FILES: tuple[tuple[str, str], ...] = (
    ("name", "self/name.md"),
    ("description", "self/description.md"),
    ("personality", "self/personality.md"),
    ("scenario", "self/scenario.md"),
    ("first_mes", "self/first_mes.md"),
    ("mes_example", "self/mes_example.md"),
    ("system_prompt", "self/system_prompt.md"),
    ("post_history_instructions", "self/post_history_instructions.md"),
    ("creator_notes", "self/creator_notes.md"),
)

_AUDIT_SOURCE = "character_card_unpacker"


class CharacterUnpackError(Exception):
    """On-disk content differs from card and ``overwrite=True`` not set."""


def unpack_character(
    store: MemoryStore,
    card: CharacterCard,
    *,
    overwrite: bool = False,
) -> list[str]:
    """Write non-empty card fields into ``self/`` directory.

    :return: Relative paths actually written (empty when no-op).
    :raises CharacterUnpackError: If on-disk content differs and
        *overwrite* is False.
    """
    written: list[str] = []
    differences: list[str] = []

    for attr, rel_path in _FIELD_FILES:
        new_value: str = getattr(card, attr) or ""
        existed, existing = _read_or_missing(store, rel_path)

        if not existed:
            # first write — always allowed
            if new_value:
                store.write_file(rel_path, new_value, source=_AUDIT_SOURCE)
                written.append(rel_path)
            continue

        if new_value == existing:
            continue

        # on-disk differs — gate on explicit flag
        if not overwrite:
            differences.append(rel_path)
            continue

        if new_value:
            store.write_file(rel_path, new_value, source=_AUDIT_SOURCE)
            written.append(rel_path)
        else:
            _delete_if_present(store, rel_path)

    if differences:
        joined = ", ".join(differences)
        msg = (
            f"Refusing to unpack character card: existing on-disk content "
            f"would change at: {joined}. Pass overwrite=True to proceed."
        )
        raise CharacterUnpackError(msg)

    return written


def _read_or_missing(store: MemoryStore, rel_path: str) -> tuple[bool, str]:
    """Return ``(existed, content)``. Distinguishes missing from empty."""
    try:
        return True, store.read_file(rel_path)
    except MemoryStoreError:
        return False, ""


def _delete_if_present(store: MemoryStore, rel_path: str) -> None:
    """Remove file via store's path resolver (traversal safety).

    Uses direct unlink — MemoryStore has no public delete yet.
    """
    resolved = store._resolve(rel_path)  # noqa: SLF001
    if resolved.exists() and resolved.is_file():
        resolved.unlink()
