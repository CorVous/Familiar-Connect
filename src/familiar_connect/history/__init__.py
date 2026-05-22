"""Persistent conversation history.

:class:`HistoryStore` — SQLite-backed turn log plus per-(guild,
familiar, channel) rolling summary cache. Text/voice loops write;
:class:`HistoryProvider` reads.

:class:`AsyncHistoryStore` dispatches SQLite calls to background
thread; keeps event loop free.
"""
