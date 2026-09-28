# Privacy Policy

**Last updated: 2026-08-16**

Familiar-Connect is a self-hosted, open-source Discord bot. This page
describes what the software records, where it puts it, and which
third-party services it sends it to. See the
[Terms of Use](terms-of-use.md) for the disclaimers.

## Who is actually handling your data

There is no Familiar-Connect service, company, or server. The project
publishes source code. Every running bot is an **instance** somebody else
installed on their own machine, with their own API keys. That person —
the **operator** — holds the database, sets the configuration, and
decides what happens to your data. The upstream project never sees it and
cannot access, export, or delete it.

The software sends **no telemetry, analytics, or crash reports** anywhere.
There is no phone-home of any kind.

So: this page tells you what the software *does*. For what a particular
bot does with what it recorded, ask whoever invited it.

## When the bot records

Only where it has been explicitly switched on:

- **Text channels** an operator ran `/subscribe-text` in. Unsubscribed
  channels are ignored entirely, including threads and forum posts, which
  count as their own channels.
- **Voice channels** the operator joined it to with `/subscribe-voice`.
- **Direct messages**, but only from users on the operator's configured
  `dm_allowlist`. On the first admitted DM the bot posts a warning that
  the conversation is not private.

Messages from other bots, and the bot's own messages, are skipped.

Bear in mind that the bot's recall is **not** scoped per channel: the
recent-history layer queries across every channel it has recorded, so
something said in one place can surface in a reply somewhere else, in
front of different people. Treat anything said near the bot as public.

## What is stored on disk

Everything lives in one SQLite database, `history.db`, inside the
familiar's folder under the familiars root — see
[On-disk layout](../getting-started/on-disk-layout.md) for the exact
paths. Two [tantivy](https://github.com/quickwit-oss/tantivy) full-text
indexes, `fts/turns/` and `fts/facts/`, sit beside it. Everything is
plaintext on the operator's filesystem; the database is not encrypted.

Recorded directly:

- **Message and transcript text, in full.** Every text message and every
  finalised voice transcript becomes a row with the channel id, guild id,
  Discord message id, reply-to id, timestamp, and the bot's own replies
  and tool calls.
- **Speaker identity.** Discord user id, username, display name, and
  platform, stored per turn and again as a durable profile row: username,
  global name, per-guild nicknames, and — when Discord exposes them —
  your **profile pronouns and bio**.
- **Reactions and mentions.** Emoji reaction counts per message, and who
  was mentioned in which turn.

Derived by feeding the above back through an LLM:

- **Facts** — short extracted statements about people, tagged with a
  subject, an importance score, and validity dates.
- **People dossiers** — a running LLM-written profile of each person the
  bot has learned about, compounded over time.
- **Summaries** and **reflections** — rolling per-channel summaries and
  higher-order syntheses citing the turns and facts they came from.
- **Opinions and dream text** generated during the nightly sleep pass.
- **Embeddings** of facts, if the operator turns that on. These are
  computed locally.

Also on disk: `subscriptions.toml` (which channels the bot listens to)
and, when configured, `character.md`, `lorebook.toml`, and
`activities.toml`, which the operator writes by hand.

## Voice audio is not recorded

Voice audio is never written to disk. Discord's stream is decoded in
memory, split per speaker, and streamed straight to the speech-to-text
provider. The only buffers are transient: a few seconds of audio held for
reconnect replay, and the current utterance held for turn detection. Both
are discarded. What persists is the **transcript text**.

Note that Discord's DAVE end-to-end encryption does not shield you from
the bot. The bot is a full member of the encrypted voice session, so it
decrypts and decodes everyone's audio in its own process. DAVE protects
that audio in transit from Discord and from non-members, not from a
participant.

Synthesized speech the bot itself produces may be cached to disk under
`data/cache/greetings`. That contains the bot's own voice reading
operator-configured greeting strings — no user audio.

## What is sent to third parties

Every one of these is the operator's account, under the operator's
agreement with that provider, and none of them are controlled by this
project. Review their policies yourself.

| Service | Receives | When |
|---|---|---|
| **Discord** | Everything, inherently — it is the transport. Plus the bot's replies, typing indicators, presence, and synthesized voice. | Always. |
| **OpenRouter** (`openrouter.ai`) | The assembled prompt: recent conversation across channels, summaries, dossiers, facts, reflections, your Discord display name, a `discord_<user id>` identifier, and any images. OpenRouter then routes it to whichever model provider the operator picked, so it reaches that provider too. | Every reply, plus every background memory pass (fact extraction, dossiers, summaries, reflections, sleep). |
| **Deepgram** (`api.deepgram.com`) | Raw voice audio, streamed live per speaker. The connection URL also carries the display names, usernames, and nicknames of everyone in the voice channel, as recognition hints. | Whenever the bot is in a voice channel. |
| **Cartesia** (`api.cartesia.ai`) | The text the bot is about to speak. No user names or ids — though the bot's reply can of course quote you. | Whenever the bot speaks. |
| **Image hosts on the allowlist** (Discord's CDNs plus a short list of image CDNs — `[tools].trusted_image_hosts`) | An HTTP request from the operator's machine, exposing its IP address to that host. | When the model uses `view_image` on an attachment, an embed, or a pasted URL. Hosts outside the allowlist are refused without any request being made, unless the operator sets `[tools].allow_untrusted_image_urls = true`, which permits any public host. Fetched images are then sent to OpenRouter. |

Cartesia is the only text-to-speech provider. Azure Speech and Google
Gemini were previously selectable in configuration but never had working
backends; they have been removed, and no audio or text has ever been sent
to either.

Embeddings are computed on the operator's own machine — there is no
remote embedding provider. Optional local models (the ONNX turn detector,
the local embedder) are downloaded from Hugging Face on first use; that
transfers model weights in, never your data out.

Twitch support is feature-gated, off by default, and its EventSub client
is not implemented yet.

## Retention: there isn't a policy

**Nothing expires and nothing is deleted on a schedule.** There is no
retention window, no TTL, no automatic purge. Conversation turns are
append-only and stay in the database for as long as the operator keeps
the file — which may be forever.

The few things that do get removed are bookkeeping, not deletion:

- Facts are **superseded**, never deleted. The retired row stays, marked
  with when and by what, so prior beliefs remain reconstructable.
- A person's dossier row is dropped and rebuilt from current facts when a
  source fact retires.
- Reaction rows disappear when the reaction is removed on Discord.

Backups, log captures, and copies of the database are entirely up to the
operator, and this project has no visibility into them.

## Deleting your data

**There is no user-facing deletion command.** The bot's slash commands
are `/subscribe-text`, `/unsubscribe-text`, `/subscribe-voice`,
`/unsubscribe-voice`, and `/diagnostics` — none of them erase anything.
Unsubscribing stops new recording; it does not remove what was already
recorded.

To have data removed, ask the operator. They can delete rows or drop
`history.db` outright with any SQLite client; the derived side-indexes
rebuild from what is left. Whether they do so, and how fast, is between
you and them.

Data already sent to OpenRouter, Deepgram, or Cartesia is governed by
those providers' own retention practices and cannot be recalled from
here.

## Logs

The bot logs to its console. Log lines include truncated message text,
extracted fact text, and the speech-to-text connection URL with voice
members' names in it. Whether that console output is captured to a file,
shipped somewhere, or shown on a stream is the operator's business.

## Children

Familiar-Connect is not directed at children. Use Discord's own minimum
age requirement for your region. Operators should not run instances in
spaces aimed at children.

## Changes

This policy may change without notice. The "Last updated" date above is
the only version marker.

## Not legal advice

This page was written by hobbyists, not lawyers, and is not legal advice.
If you run an instance for other people, you are the one deciding what
gets recorded and where it goes — in EU/UK terms, you are the data
controller — and you are responsible for your own compliance
obligations, including any notice, consent, retention, and deletion
duties that apply to you. The upstream project cannot assess or discharge
those for you. If that matters for your deployment, get real advice.

## Contact

Questions about the software go to the
[issue tracker](https://github.com/CorVous/Familiar-Connect/issues).
Questions about a specific bot's data go to whoever runs it.
