# Slash commands

Once the bot is online and invited to a test guild, the subscription
surface is:

<!-- @slash-commands-table -->

Table above is generated at build time from `bot.slash_command(...)`
calls in `src/familiar_connect/bot.py` — edit the source to change a
description and this page updates automatically.

Subscriptions persist across restarts at
`data/familiars/<id>/subscriptions.toml`; delete that file to reset.

Threads and forum posts are `discord.Thread` instances: running
`/subscribe-text` inside a thread subscribes the familiar to that
thread only.
