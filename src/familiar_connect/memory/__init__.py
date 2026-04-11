"""Per-familiar memory layer.

The familiar's memory is a directory of plain-text files on disk
(see docs/architecture/memory.md).
This package owns the on-disk shape and the safe file-IO surface
the rest of the bot — and, eventually, a tool-using cheap model —
uses to read and write that directory.

One-shot operator utilities that seed a store from external assets
(character-card unpacker, SillyTavern lorebook importer) live in
:mod:`familiar_connect.bootstrap`, not here; this package is strictly
the runtime file-IO surface.
"""
