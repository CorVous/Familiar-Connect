# Slash commands

Once the bot is online and invited to a test guild, the subscription
surface is:

| Command | What it does |
|---|---|
| `/subscribe-text` | Listen for text messages in this channel. |
| `/unsubscribe-text` | Stop listening for text messages in this channel. |
| `/diagnostics` | Show span timings (last p50/p95 per span). |
| `/subscribe-voice` | Join your voice channel and listen. |
| `/unsubscribe-voice` | Leave the voice channel in this guild. |

The commands are registered in `familiar-connect/src/bot.rs`; the two
voice commands are only registered in a `discord-voice` build.

Subscriptions persist across restarts at `subscriptions.toml` in the
familiar's data folder (see [On-disk layout](on-disk-layout.md)); delete
that file to reset.

Discord threads and forum posts are their own channels: running
`/subscribe-text` inside a thread subscribes the familiar to that
thread only.
