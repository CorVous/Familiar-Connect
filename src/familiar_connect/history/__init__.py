"""Persistent conversation history for the bot.

The :class:`HistoryStore` is the SQLite-backed record of every
conversational turn the bot sees, plus a per-(guild, familiar,
channel) cache of rolling summaries built from older turns by a
cheap side-model. The bot's text-session and voice-session loops
write turns into it; the context pipeline's
:class:`HistoryProvider` reads from it.

:class:`AsyncHistoryStore` wraps :class:`HistoryStore` so that all
SQLite calls run on a dedicated background thread rather than blocking
the event loop.
"""
