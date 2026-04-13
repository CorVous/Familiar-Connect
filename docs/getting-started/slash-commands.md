# Slash commands

Once the bot is online and invited to your test guild, the subscription
surface is:

<!-- @slash-commands-table -->

The table above is generated at build time from
`bot.slash_command(...)` calls in `src/familiar_connect/bot.py` — edit
the source to change a description and this page updates automatically.

The channel-mode commands map to config names: `/channel-full-rp` sets
`full_rp` (all providers on), `/channel-text-conversation-rp` sets
`text_conversation_rp`, and `/channel-imitate-voice` sets
`imitate_voice` (tight budget, low TTFB).

A voice channel first reached via `/subscribe-my-voice` defaults to
`imitate_voice` automatically — there's no need to also run
`/channel-imitate-voice` unless you later want to switch profiles.
Text channels without a sidecar fall back to the character's
`default_mode` instead.

See also the [Twitch guide](../guides/twitch.md) for the `/twitch *`
command surface.

## Full end-to-end smoke test

1. `uv run familiar-connect -vv run --familiar aria`
2. In a test text channel, run `/subscribe-text`. Send a message — the
   bot should reply.
3. Run `/channel-full-rp` and send another — the log line
   `pipeline channel=… provider=… status=ok duration=…` should now
   show `content_search` running alongside `character` and `history`.
4. Join a voice channel and run `/subscribe-my-voice`. The bot should
   join and greet. Post another message in the subscribed text
   channel — the reply should also be spoken.
5. `/unsubscribe-voice` then `/unsubscribe-text` to tear down cleanly.

Subscriptions and channel modes persist across restarts under
`data/familiars/<id>/subscriptions.toml` and
`data/familiars/<id>/channels/<channel_id>.toml`; delete those files
to reset.

## Voice input caveat

`/subscribe-my-voice` currently joins the voice channel and keeps a
PCM sink open for TTS replies, but **incoming audio is not yet wired
into the reply pipeline**. See [Voice input](../roadmap/voice-input.md)
for the roadmap entry.
