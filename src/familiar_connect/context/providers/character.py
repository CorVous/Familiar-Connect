"""CharacterProvider — surfaces the familiar's ``self/`` files.

Step 5 of future-features/context-management.md. Reads every Markdown
file directly inside ``self/`` in the familiar's :class:`MemoryStore`
and emits one :class:`Contribution` per non-empty file at high
priority on :data:`Layer.character`. The provider is always on — if
the familiar has no character description, it returns an empty list,
which the budgeter and pipeline handle naturally.

The provider is bound to a single :class:`MemoryStore` at
construction. The orchestrating bot is responsible for instantiating
one provider per familiar and registering it with the per-familiar
:class:`ContextPipeline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer
from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest
    from familiar_connect.memory.store import MemoryStore


CHARACTER_PRIORITY = 100
"""Priority assigned to every CharacterProvider contribution.

Higher than the default for retrieved/searched content so the
familiar's persona always survives token-budget pressure ahead of
arbitrary memory snippets."""

_DEADLINE_S = 0.25
"""Deadline for the provider's contribute() call.

Filesystem reads against a small ``self/`` directory should complete
in milliseconds; this is a generous upper bound."""

_SELF_DIR = "self"


class CharacterProvider:
    """ContextProvider that surfaces files under ``self/``.

    Conforms to the ContextProvider Protocol structurally — no
    inheritance required.
    """

    id = "character"
    deadline_s = _DEADLINE_S

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def contribute(
        self,
        request: ContextRequest,  # noqa: ARG002
    ) -> list[Contribution]:
        """Return one Contribution per non-empty Markdown file in ``self/``.

        - Empty files are skipped.
        - Files whose name doesn't end in ``.md`` are skipped.
        - Subdirectories under ``self/`` are skipped — ``self/`` is a
          flat namespace for the unpacked card fields.
        - A missing ``self/`` directory returns an empty list.
        """
        try:
            entries = self._store.list_dir(_SELF_DIR)
        except MemoryStoreError:
            return []

        contributions: list[Contribution] = []
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.is_dir:
                continue
            if not entry.name.endswith(".md"):
                continue

            rel = f"{_SELF_DIR}/{entry.name}"
            try:
                text = self._store.read_file(rel)
            except MemoryStoreError:
                continue
            if not text:
                continue

            contributions.append(
                Contribution(
                    layer=Layer.character,
                    priority=CHARACTER_PRIORITY,
                    text=text,
                    estimated_tokens=estimate_tokens(text),
                    source=f"character:{entry.name.removesuffix('.md')}",
                )
            )

        return contributions
