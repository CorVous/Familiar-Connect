# Slash commands

Once the bot is online and invited to your test guild, the subscription
surface is:

| Command | What it does |
|---|---|
| `/subscribe-text` | Listen for messages in the current text channel |
| `/unsubscribe-text` | Stop listening in the current text channel |
| `/subscribe-my-voice` | Join your voice channel (if you're in one) and enable TTS replies |
| `/unsubscribe-voice` | Leave the voice channel |
| `/channel-full-rp` | Put the current channel into `full_rp` mode (all providers on) |
| `/channel-text-conversation-rp` | Put the current channel into `text_conversation_rp` mode |
| `/channel-imitate-voice` | Put the current channel into `imitate_voice` mode (tight budget, low TTFB) |

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
