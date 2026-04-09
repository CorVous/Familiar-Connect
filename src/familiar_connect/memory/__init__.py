"""Per-familiar memory layer.

The familiar's memory is a directory of plain-text files on disk
(see plan.md § Memory Directory and future-features/memory.md).
This package owns the on-disk shape and the safe file-IO surface
the rest of the bot — and, eventually, a tool-using cheap model —
uses to read and write that directory.
"""
