# Bootstrapping a familiar

One-shot operator utilities for seeding a familiar from external
assets. **Nothing in this file is invoked by the bot at runtime.** If
you are only running the bot against a familiar whose `memory/`
directory is already populated, you can ignore this document entirely.

Everything listed here lives under `familiar_connect.bootstrap` in the
source tree and is expected to be called by hand (Python REPL,
one-off script, or future CLI subcommand) when you are setting up a
new familiar. The dependency direction is one-way: the `bootstrap`
package imports from `familiar_connect.memory.store`, but nothing in
the runtime reply pipeline (`bot.py`, `familiar.py`, `commands/run.py`)
ever imports from `bootstrap`. That invariant is enforced by a ruff
`flake8-tidy-imports` `banned-api` rule in `pyproject.toml` — any PR
that accidentally crosses the boundary will fail lint.

## Prerequisites

- A familiar directory on disk at `data/familiars/<id>/`. The smallest
  layout that boots is documented in
  [On-disk layout](../getting-started/on-disk-layout.md); the bootstrap
  utilities write into `data/familiars/<id>/memory/` below that root.
- At least one of:
    - A SillyTavern Character Card V3 PNG (for the unpacker), or
    - A SillyTavern lorebook / world-info JSON export (for the
      importer).

## Unpacking a character card

Translates a V3 character card's non-empty fields into one Markdown
file each under `self/` in the familiar's `MemoryStore`. The
`CharacterProvider` then surfaces those files per turn at runtime, so
after this runs once the card is fully integrated into the familiar's
persona. The unpacker is **idempotent** — re-running with the same
card is a no-op — and gated: re-running with a *different* card
errors unless you pass `overwrite=True`. See the
[Context pipeline](../architecture/context-pipeline.md) step 4 for
the design.

```python
from pathlib import Path

from familiar_connect.bootstrap.unpack_character import unpack_character
from familiar_connect.character import load_card
from familiar_connect.memory.store import MemoryStore

card = load_card(Path("aria-v3.png"))
store = MemoryStore(Path("data/familiars/aria/memory"))

written = unpack_character(store, card)
print("wrote:", written)
```

Files produced (empty fields are omitted — there is never an empty
placeholder on disk):

- `self/name.md`
- `self/description.md`
- `self/personality.md`
- `self/scenario.md`
- `self/first_mes.md`
- `self/mes_example.md`
- `self/system_prompt.md`
- `self/post_history_instructions.md`
- `self/creator_notes.md`

To re-unpack a card whose contents have changed, pass
`overwrite=True`. Under `overwrite`, only the fields that actually
changed are rewritten — and a field that has become empty is removed
from disk so the on-disk shape always reflects the current card.

## Importing a SillyTavern lorebook

Translates a SillyTavern lorebook / world-info JSON export into one
Markdown file per entry under `lore/imported/` in the familiar's
`MemoryStore`. Each file is plain Markdown — an H1 built from the
entry's comment, the trigger keywords in a blockquoted bullet list at
the top (kept for human reference only, **not** used at runtime), and
the entry body as the content. Once imported, the files are
indistinguishable from any other Markdown in the memory directory and
the agentic `ContentSearchProvider` finds them via grep like anything
else. There is no runtime keyword walker, no World Info trigger
logic, and no special data path — the import is a one-shot
translation, not an ongoing dependency. See the
[Context pipeline](../architecture/context-pipeline.md) step 9 for
the design.

```python
from pathlib import Path

from familiar_connect.bootstrap.import_silly_tavern import (
    import_silly_tavern_lorebook,
)
from familiar_connect.memory.store import MemoryStore

store = MemoryStore(Path("data/familiars/aria/memory"))

result = import_silly_tavern_lorebook(store, Path("lorebook.json"))
print("written:", result.written)
print("skipped:", result.skipped)
print("errors:", result.errors)
```

Options:

- `target_dir="lore/imported"` — relative subdirectory under the store
  root to write the imported files into. Must not escape the store
  root.
- `force=False` — by default, files that already exist at the
  destination path are left untouched and recorded in
  `result.skipped`. Pass `force=True` to overwrite them.

The importer is non-fatal at the entry level: malformed entries,
oversized entries, and entries that would collide with existing files
(without `force=True`) are recorded in the returned `ImportResult`
rather than aborting the whole import. Top-level errors (unreadable
file, invalid JSON, missing `entries` field) raise
`LorebookImportError`.

## Why these utilities are quarantined

These tools write Markdown into a `MemoryStore` once, at setup time,
and are then never touched again. Keeping them in their own
subpackage means:

- The runtime's import graph stays small and auditable. No operator
  glue code is loaded when the bot starts.
- Engineers (and AI assistants) reading `src/familiar_connect/memory/`
  see only the hot-path file-IO surface, not the one-shot converters.
- Bit-rot is caught by the standard pytest run — the bootstrap tests
  live under `tests/bootstrap/` and run in every CI invocation.
- Accidental coupling is caught at lint time by the ruff `TID251`
  rule, which bans imports of `familiar_connect.bootstrap` outside
  the bootstrap package and its tests.

## Future: CLI subcommand

A `familiar init --from-card` CLI subcommand is on the roadmap but
deferred; see the
[Context pipeline](../architecture/context-pipeline.md) for details.
Until that lands, the programmatic recipes above are the supported
operator interface.
