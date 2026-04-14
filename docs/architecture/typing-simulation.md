# Typing simulation

For text-RP channels, the familiar's reply does not land as a single wall of
text. Instead, the LLM output is split into paragraph-sized chunks, and each
chunk is preceded by a `typing…` indicator whose duration is proportional to
the chunk length. This produces a "the familiar is typing in real time" feel
that matches how a human would compose a longer message.

A new user message arriving mid-delivery **cancels** the remaining chunks.
What was already sent is persisted to history as an assistant turn; the new
message flows through the conversation monitor normally and drives a fresh
reply that sees both the partial previous reply and the new context.

!!! note "Scope"
    Typing simulation is default-on for `text_conversation_rp` channels,
    default-off for `full_rp` and `imitate_voice`. Operators can override
    per familiar or per channel. See [Configuration](#configuration) below.

---

## Flow

```
[pipeline → LLM → post-processing]
  (existing typing… wraps this whole "thinking" phase)

[chunk 1]  typing…  (delay ∝ len(chunk)) → channel.send(chunk)
           inter-line pause
[chunk 2]  typing…  → channel.send(chunk)
           inter-line pause
...
```

1. The existing `async with channel.typing():` that wrapped pipeline
   assembly + LLM + post-processing still shows while the familiar is
   "thinking".
2. After the LLM produces `reply_text`, `split_reply_into_chunks` breaks
   it into paragraph-sized pieces.
3. For each chunk: enter `channel.typing()`, `asyncio.sleep(delay)`,
   `channel.send(chunk)`, `asyncio.sleep(inter_line_pause_s)`, loop.
4. When the loop finishes, the assistant turn is written to history and
   TTS fan-out runs (if subscribed in the same guild).

### Cancellation

`on_message` consults the per-channel `TextDeliveryTracker` before
routing to the conversation monitor. If a delivery task is still
running, it is cancelled and awaited:

```python
tracker = familiar.text_delivery_registry.get(channel_id)
if tracker.is_active():
    await tracker.cancel_and_wait()
```

The task raises `CancelledError`, which the response handler catches.
Whatever chunks were already delivered to Discord are persisted as the
assistant turn; unsent chunks are discarded. TTS fan-out is skipped for
cancelled deliveries — the new user message will drive a fresh reply
through the monitor, and that fresh reply is what gets spoken.

The history ordering for a cancel scenario is:

1. User turns from the original buffer (persisted **before** delivery).
2. Partial assistant turn (only chunks actually sent).
3. New user message (via monitor's own buffer).
4. Fresh assistant turn (from the new LLM call).

---

## Chunk splitting

`familiar_connect.text.delivery.split_reply_into_chunks` splits in two
passes:

1. **Paragraphs.** Blank-line-separated paragraphs are the primary unit.
2. **Sentence fallback.** Any paragraph longer than
   `sentence_split_threshold` characters is further split on sentence
   terminators (`. ! ? …` followed by whitespace).
3. **Hard cap.** As a last resort, any chunk still over 1900 characters
   is split at the last whitespace before the cap, to stay under
   Discord's 2000-character message limit.

Leading and trailing whitespace is stripped per chunk; empty chunks are
dropped.

The sentence splitter is a simple regex — it does not know about
abbreviations like `Dr.` or `U.S.`. For conversational RP prose this is
acceptable; contributions welcome for a smarter splitter.

---

## Typing delay

`familiar_connect.text.delivery.compute_typing_delay` returns:

```
raw = len(chunk) / chars_per_second
delay = clamp(raw, min_delay_s, max_delay_s)
```

If `chars_per_second <= 0`, the delay returns `max_delay_s` (safety fallback
so a misconfigured profile doesn't break delivery).

---

## Configuration

Per-channel `typing_simulation` settings are layered from three sources,
applied in order:

1. **Per-mode default** in `channel_config_for_mode` (lowest priority).
   `text_conversation_rp` sets `enabled=True`; `full_rp` and
   `imitate_voice` set `enabled=False`.
2. **Character-level overrides** from `character.toml`
   `[typing_simulation]`.
3. **Channel-level overrides** from
   `channels/<channel_id>.toml` `[typing_simulation]` (highest priority).

Each layer is a partial override: only fields the user explicitly sets in
TOML substitute the lower-priority value.

### `character.toml`

```toml
[typing_simulation]
# Master on/off — inherits from mode default when omitted.
enabled = true

# Typing speed in characters per second (≈ 40 cps → 240 wpm).
chars_per_second = 40.0

# Floor and ceiling for the per-chunk typing delay.
min_delay_s = 0.8
max_delay_s = 6.0

# Pause between sending a chunk and showing "typing…" for the next chunk.
# No typing indicator is shown during this pause.
inter_line_pause_s = 0.7

# Paragraphs longer than this (characters) split by sentence.
sentence_split_threshold = 400
```

### `channels/<channel_id>.toml`

```toml
mode = "text_conversation_rp"

# Override just the typing speed for this channel.
[typing_simulation]
chars_per_second = 25.0
```

### Disabling for one channel

```toml
mode = "text_conversation_rp"

[typing_simulation]
enabled = false
```

---

## Implementation modules

| Piece | Module |
|---|---|
| Dataclass + per-mode defaults | `familiar_connect.config.TypingSimulationConfig`, `channel_config_for_mode` |
| Partial-override parser | `familiar_connect.config._parse_typing_simulation_overrides` |
| Chunk splitter + delay math | `familiar_connect.text.delivery.split_reply_into_chunks`, `compute_typing_delay` |
| Per-channel in-flight tracker | `familiar_connect.text.delivery.TextDeliveryTracker`, `TextDeliveryRegistry` |
| Chunked delivery loop | `familiar_connect.bot._run_text_response` |
| Cancellation hook | `familiar_connect.bot.on_message` |

The tracker pattern deliberately mirrors `ResponseTracker` /
`ResponseTrackerRegistry` in `familiar_connect.voice.interruption` — the
text path now has its own state-tracking surface for mid-flight cancel,
analogous to the voice path's long-burst-at-`GENERATING` case. See
[Voice interruption](interruption.md) for the counterpart on the voice
side.
