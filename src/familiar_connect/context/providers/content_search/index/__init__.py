"""Derived-index subpackage — SQLite-backed embedding cache + write hooks.

The index under ``memory/.index/`` is a regenerable cache of the
canonical Markdown files. It may be deleted at any time; the next
startup rebuilds it. See docs/architecture/memory.md "Derived
indices".
"""
