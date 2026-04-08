"""Unpack a CharacterCard into a MemoryStore's ``self/`` directory.

Step 4 of future-features/context-management.md. On familiar creation
the loaded :class:`~familiar_connect.character.CharacterCard` is
walked field by field; each non-empty field becomes a Markdown file
under ``self/`` in the familiar's :class:`MemoryStore`. Empty fields
produce no file at all — there is never an empty placeholder on disk.

The unpacker is **idempotent**: re-unpacking the same card is a
no-op. Re-unpacking a card whose contents differ from what is already
on disk raises :class:`CharacterUnpackError` unless the caller passes
``overwrite=True``. Under overwrite, only the fields that actually
changed are rewritten — and a field that has become empty is removed
from disk so the on-disk shape always reflects the current card.

The unpacker writes through the store, so every change goes through
the store's path-traversal safety, atomic write, and audit log
machinery automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.character import CharacterCard
    from familiar_connect.memory.store import MemoryStore


# Mapping from CharacterCard attribute name to its filename under ``self/``.
# Order is the canonical order we walk the card and the order in which
# returned paths appear, so callers (and tests) get a predictable list.
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
    """Raised when an unpack would change existing on-disk content.

    Specifically: a field's on-disk content differs from the value in
    the new card, and the caller did not pass ``overwrite=True``.
    """


def unpack_character(
    store: MemoryStore,
    card: CharacterCard,
    *,
    overwrite: bool = False,
) -> list[str]:
    """Write *card*'s non-empty fields into ``store``'s ``self/`` directory.

    :param store: The familiar's :class:`MemoryStore`.
    :param card: A loaded :class:`CharacterCard`.
    :param overwrite: If ``True``, replace differing on-disk content
        and remove files for fields that have become empty. If
        ``False`` (the default) and the card differs from what's on
        disk, raise :class:`CharacterUnpackError`.
    :return: List of relative paths that were actually written this
        call. Empty when the unpack was a no-op.
    :raises CharacterUnpackError: If a difference is detected and
        ``overwrite`` is ``False``.
    """
    written: list[str] = []
    differences: list[str] = []

    for attr, rel_path in _FIELD_FILES:
        new_value: str = getattr(card, attr) or ""
        existed, existing = _read_or_missing(store, rel_path)

        if not existed:
            # First write of this field. Always allowed; nothing to overwrite.
            if new_value:
                store.write_file(rel_path, new_value, source=_AUDIT_SOURCE)
                written.append(rel_path)
            continue

        if new_value == existing:
            # On-disk content already matches the card. No-op.
            continue

        # On-disk content disagrees with the card. This is an
        # overwrite — gate it on the explicit flag.
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
    """Return ``(existed, content)`` for *rel_path* in *store*.

    Distinguishes "file doesn't exist" from "file exists but is empty",
    so the caller can tell a first write from an overwrite.
    """
    try:
        return True, store.read_file(rel_path)
    except MemoryStoreError:
        return False, ""


def _delete_if_present(store: MemoryStore, rel_path: str) -> None:
    """Remove a file inside the store, if it exists.

    Goes through the store's path resolver so traversal safety still
    applies. Currently uses a direct unlink because the public
    :class:`MemoryStore` API doesn't expose a delete method yet —
    that's a small, local extension we may want to land separately.
    """
    # Use the store's resolver so the path is validated against the root.
    resolved = store._resolve(rel_path)  # noqa: SLF001
    if resolved.exists() and resolved.is_file():
        resolved.unlink()
