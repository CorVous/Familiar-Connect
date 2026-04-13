# Conversation flow: chattiness & interjection

How the familiar decides **whether** and **when** to speak in a conversation it hasn't been directly addressed in.

!!! info "Status: Design"
    Not yet shipped. Today the bot responds to **every** message on a subscribed channel — `bot.py:on_message` goes straight from subscription check to context pipeline to LLM call. There is no decision gate. This page describes the gate we want to build.

## Motivation

Two orthogonal controls should govern the familiar's conversational behaviour:

- **Chattiness** — a free-text personality trait that shapes *how willing* the familiar is to respond. Fed directly into a side-model evaluation prompt. No enum, no tiers — the LLM interprets the personality naturally.
- **Interjection** — a 5-tier enum that controls *how patient* the familiar is about waiting for a lull before evaluating. Governs the mechanical timing of when the side model is consulted, plus prompt tone.

A third setting, **lull timeout**, controls how long a gap in messages counts as "the conversation has paused."

### Behaviour matrix

|                      | Low interjection                                 | High interjection                                |
|----------------------|--------------------------------------------------|--------------------------------------------------|
| **Reserved chattiness** | Speaks rarely, waits for silence                 | Speaks rarely, but when it does, cuts right in   |
| **Eager chattiness**    | Wants to respond to everything, but waits politely | Won't shut up and talks over people              |

### Why not just the preset?

An earlier sketch suggested a "chattiness preset" governing response frequency alone. That misses the character layer: a familiar set to "moderate" should feel very different depending on who it is. A shy familiar hedges and apologises when it speaks up unprompted; an arrogant one acts like it's doing the room a favour; a curious one opens with a question. The preset governs *when* it responds. The personality description governs *how it feels* about responding. This design keeps both: `chattiness` for personality, `interjection` for mechanical timing.

## Configuration

All three settings live in `character.toml` on `CharacterConfig`:

```toml
aliases = ["aria", "ari"]
chattiness = "Curious and opinionated, but knows when to let others have their moment"
interjection = "average"
text_lull_timeout = 10.0
voice_lull_timeout = 2.0
```

### `aliases: list[str]`

Additional names the familiar responds to beyond the `familiar_id` (the folder name). Used for direct-address detection alongside Discord @mentions.

**Default:** `[]` (only the familiar's folder name and @mentions trigger direct address).

### `chattiness: str`

Free-text personality trait describing the familiar's conversational disposition. Injected verbatim into the side-model evaluation prompt. The LLM decides whether the familiar would want to respond based on this personality, the conversation content, and the trigger context.

**Default:** `"Balanced — responds when the conversation is relevant"`.

**Examples:**

- `"Shy and reserved, only speaks when they have something truly meaningful to add"`
- `"Curious and opinionated, always has a take on everything"`
- `"Playful and easily excited, loves jumping into conversations about games"`

### `interjection: Interjection`

Enum controlling how long the familiar waits before the side model is even consulted during an active conversation. Higher values mean the familiar evaluates sooner and more frequently.

**Default:** `average`

| Value                | Starting interval (messages) |
|----------------------|------------------------------|
| `very_quiet`         | 15                           |
| `quiet`              | 12                           |
| `average`            | 9                            |
| `eager`              | 6                            |
| `very_eager`         | 3                            |

### `text_lull_timeout: float`

Seconds of silence on a **text** channel before the lull evaluation fires. Text-only — the voice path has its own debounce (`voice_lull_timeout`). Configurable so operators can tune per-familiar.

**Default:** `10.0`

### `voice_lull_timeout: float`

Seconds of channel-wide silence after which a buffered voice utterance is handed to the response pipeline. Acts as a debounce on Deepgram's per-final transcript stream so the bot only responds once the speaker has actually paused, rather than on every mid-sentence fragment.

Silence is detected Discord-natively: every inbound audio frame routed through the voice pipeline resets per-user silence watchdogs inside `VoiceLullMonitor` (`src/familiar_connect/voice_lull.py`). When every user has been quiet for longer than `user_silence_s` (200 ms) *and* there is at least one buffered final, the lull timer starts; on expiry the concatenated text is sent to `_handle_voice_result` via `on_utterance_complete`. Unlike the text lull, there is no side-model yes/no gate — voice lull is pure debounce.

**Default:** `2.0`

## Sketch

### `ConversationMonitor`

A standalone class that owns all per-channel conversation state and is the single entry point `bot.py` calls instead of going straight to the pipeline. Held on the `Familiar` bundle, built in `Familiar.load_from_disk`.

```python
class ConversationMonitor:
    def __init__(
        self,
        familiar_name: str,
        aliases: list[str],
        chattiness: str,
        interjection: Interjection,
        lull_timeout: float,
        side_model: SideModel,
        character_card: str,
        on_respond: Callable[[int, list[BufferedMessage]], Awaitable[None]],
    ):
        ...

    async def on_message(
        self,
        channel_id: int,
        speaker: str,
        text: str,
        is_mention: bool,
    ) -> None:
        """Called by bot.py for every message on a subscribed channel."""
        ...
```

`on_respond` is a callback that triggers the full pipeline. The monitor passes the channel ID and the buffer contents so the pipeline has the full conversation context. The callback is a closure built in `bot.py` that captures the channel/guild references needed to send the reply.

### Per-channel state: `ChannelBuffer`

The monitor holds a `dict[int, ChannelBuffer]` keyed by channel ID. Each `ChannelBuffer` contains:

- **`buffer: list[BufferedMessage]`** — messages accumulated since the bot last responded (or last cleared). Each entry is speaker + text + timestamp.
- **`message_counter: int`** — count of messages since last response. Drives the interjection check schedule.
- **`check_count: int`** — how many interjection checks have been made since the last response. Drives the step-down curve.
- **`lull_timer_handle: asyncio.TimerHandle | None`** — handle to the pending lull callback, cancelled and reset on every new message.

## Three triggers, one evaluation path

Every incoming message on a subscribed channel flows through the monitor. There are three conditions that trigger a side-model evaluation. All three use the **same** side-model call with the same core inputs — they differ only in when they fire and what extra context is included in the prompt.

### 1. Direct address (immediate)

**When:** A message contains the familiar's name, any alias (case-insensitive, word-boundary aware), or an @mention (`is_mention=True`).

**Timing:** Immediate — no waiting for lull or counter threshold.

**Extra prompt context:** `"You were directly addressed in the conversation."`

**Name matching rules:**

- Case-insensitive.
- Word-boundary aware: `"aria"` matches `"Hey Aria, what do you think?"` but not `"malaria is spreading"`.
- Matches against: `familiar_id` (folder name) + all entries in `aliases`.

### 2. Interjection check (counter-based)

**When:** The message counter reaches the current interjection threshold.

**Timing:** Threshold starts at the tier's starting interval and decreases by 3 after each check, flooring at 3.

**Extra prompt context:** `"{N} messages have been said without you speaking."` (where N is the total `message_counter` value, not the interval).

**Step-down curve.** The starting interval is tier-dependent. After each check where the familiar declines, the interval for the *next* check decreases by 3, with a floor of 3. The total message counter never resets until the familiar responds.

Example for `average` (starting interval = 9):

| Check # | Fires at message | Interval used | Next interval |
|---------|------------------|---------------|---------------|
| 1       | 9                | 9             | 6             |
| 2       | 15               | 6             | 3             |
| 3       | 18               | 3             | 3 (floor)     |
| 4       | 21               | 3             | 3             |
| ...     | every 3 after    | 3             | 3             |

Example for `very_quiet` (starting interval = 15):

| Check # | Fires at message | Interval used | Next interval |
|---------|------------------|---------------|---------------|
| 1       | 15               | 15            | 12            |
| 2       | 27               | 12            | 9             |
| 3       | 36               | 9             | 6             |
| 4       | 42               | 6             | 3             |
| 5       | 45               | 3             | 3 (floor)     |
| ...     | every 3 after    | 3             | 3             |

The prompt pressure also builds naturally: by check 5 on a `very_quiet` familiar, the prompt says *"45 messages have been said without you speaking"* — making even a reserved personality more likely to respond.

### 3. Lull (silence-based)

**When:** No new message arrives for `text_lull_timeout` seconds after the most recent message.

**Timing:** A timer is started (or reset) on every incoming message. If it expires without a new message arriving, the evaluation fires.

**Extra prompt context:** None. The lull evaluation is the baseline — just the conversation content and the familiar's personality.

### On any successful response

- Buffer is cleared.
- Message counter resets to 0.
- Check count resets to 0 (so the step-down curve restarts from the tier's starting interval).
- Lull timer is cancelled.

### On direct address (regardless of evaluation result)

Same resets as a successful response: buffer cleared, counter reset, check count reset, lull timer cancelled.

### Concurrency guard

If a lull evaluation and an interjection evaluation both trigger close together, or if a direct-address message arrives while an evaluation is in flight, only one evaluation should run at a time per channel. A per-channel `asyncio.Lock` prevents racing into two simultaneous responses.

## Side-model evaluation

### Inputs (same for all three triggers)

1. **Character card** — the full text of the familiar's character (from `memory/self/` files, pre-loaded).
2. **Conversation summary** — the rolling summary from `HistoryProvider` (the compressed view of older conversation history).
3. **Buffer contents** — the recent messages that have accumulated since the last response.
4. **Chattiness personality** — the free-text `chattiness` value from config.
5. **Trigger context** — the extra prompt context specific to the trigger (absent for lull evaluations).

### Output

The side model returns YES or NO. On YES, the `on_respond` callback fires with the buffer contents, triggering the full context pipeline → main LLM → reply → TTS flow. On NO, the monitor does nothing (except advance the step-down curve for interjection checks).

### Prompt templates (draft — configurable later)

**Lull evaluation (no extra context):**

```
You are {familiar_name}.

{character_card_summary}

Your conversational personality: {chattiness}

Here is a summary of the recent conversation:
{conversation_summary}

The following messages were just said:
{buffer}

Would you like to respond to this conversation? Answer YES or NO.
```

**Direct address:** same prompt with the line `"You were directly addressed in the conversation. Would you like to respond? Answer YES or NO."` at the end.

**Interjection:** same prompt with the line `"{message_count} messages have been said without you speaking. Would you like to interject? Answer YES or NO."` at the end.

These are starting points. The wording, structure, and whether to allow the model to explain its reasoning (chain-of-thought before the YES/NO) are all tuneable later.

## Message flow

```
message arrives on subscribed channel
    │
    ├─ bot ignores own messages (existing)
    ├─ subscription check (existing)
    │
    ▼
familiar.monitor.on_message(channel_id, speaker, text, is_mention)
    │
    ├─ cancel existing lull timer
    ├─ append message to channel buffer
    ├─ increment message counter
    │
    ├─ is direct address? (name/alias in text OR is_mention)
    │   └─ YES → acquire lock → side model eval → respond or not → reset state
    │
    ├─ counter hits interjection threshold?
    │   └─ YES → acquire lock → side model eval → respond or not
    │           └─ if NO: advance step-down curve (lower next threshold, floor at 3)
    │
    └─ start new lull timer (text_lull_timeout seconds)
            └─ on expiry → acquire lock → side model eval → respond or not
```

## Changes by file

| File | Change |
|---|---|
| `config.py` | Add `chattiness: str`, `interjection: Interjection`, `lull_timeout: float`, `aliases: list[str]` to `CharacterConfig`. Add `Interjection` enum. Load/validate from TOML. |
| `chattiness.py` | **New.** `ConversationMonitor`, `ChannelBuffer`, `BufferedMessage`, `is_direct_address()`, interjection step-down logic, lull timer management. |
| `familiar.py` | Add `monitor: ConversationMonitor` field. Build it in `load_from_disk` from config + side model + character card text. |
| `bot.py` | `on_message` calls `monitor.on_message()` instead of the pipeline. Pipeline/LLM/reply logic moves into `on_respond` callback. Voice handler gets the same treatment. Unsubscribe commands clear monitor state. |
| `tests/test_chattiness.py` | **New.** Tests for `is_direct_address`, step-down curve, lull timer firing, evaluation trigger conditions, buffer/counter reset on response. |
| `tests/test_config.py` | Tests for loading `chattiness`, `interjection`, `lull_timeout`, `aliases` from TOML. Validation of enum values and types. |
| `tests/test_bot_message_loop.py` | Adjust existing tests — `on_message` no longer calls the pipeline directly. Add cases for the monitor integration. |

## Implementation order (TDD)

Each step follows red/green: write a failing test first, then the minimum code to pass.

1. **Config fields** — `aliases`, `chattiness`, `interjection`, `lull_timeout` on `CharacterConfig` + `Interjection` enum + TOML loading/validation.
2. **`is_direct_address()`** — pure function, word-boundary-aware, case-insensitive name/alias matching.
3. **`ChannelBuffer` and `BufferedMessage`** — the per-channel state dataclasses.
4. **Interjection step-down curve** — a function that computes the next check threshold given the tier and check count.
5. **`ConversationMonitor` core** — buffer management, counter incrementing, direct-address detection triggering.
6. **Lull timer** — start/reset/expiry logic using `asyncio` call-later handles.
7. **Side-model evaluation wiring** — the monitor calls `side_model.complete()` with the assembled prompt, parses YES/NO.
8. **`on_respond` integration** — callback invocation on YES, state reset.
9. **`bot.py` integration** — replace direct pipeline calls with `monitor.on_message()`, build the `on_respond` callback, wire up cleanup on unsubscribe.
10. **`familiar.py` integration** — build the monitor in `load_from_disk`, pass through config and dependencies.

## Non-goals

- **Interruption of a reply already in progress.** This page covers the **pre-speech** decision (whether to start talking). Cutting off a reply mid-stream when someone else starts talking lives in [Voice input](voice-input.md).
- **Silence-initiated conversation starts.** The lull trigger requires *some* messages in the buffer. A feature where prolonged silence with an empty buffer causes the familiar to proactively start a new conversation is architecturally different and deferred.
- **Per-modality evaluation prompts.** Text and voice now have separate lull timeouts (`text_lull_timeout` / `voice_lull_timeout`), but they still share the same evaluation prompt structure. Tuning prompts per modality is a future addition.

## Open questions

- **Twitch events.** Twitch events (subs, bits, raids) should always be acknowledged. They could bypass the monitor entirely, or be treated as direct-address-equivalent triggers. Deferred until Twitch integration is wired into the monitor.
- **Conversation summary source.** The evaluation prompt includes "a summary of the recent conversation." At evaluation time, this could come from `HistoryProvider`'s cached rolling summary, or be generated fresh. Cached is cheaper; fresh is more accurate. Start with cached.
- **Static thresholds feel robotic.** Interjection checks fire at fixed intervals (9, 6, 3, …), so a patient observer could count messages and know exactly when the familiar is about to weigh in. A small random jitter (±1–2 messages, clamped at 1, floor at 3) would break the mechanical regularity without changing tier semantics.
