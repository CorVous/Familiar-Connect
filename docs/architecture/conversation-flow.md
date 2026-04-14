# Conversation flow: chattiness & interjection

How the familiar decides **whether** and **when** to speak in a conversation it hasn't been directly addressed in.

The monitor is wired for both **text** and **voice** channels. `bot.py:on_message` routes text messages through `familiar.monitor.on_message`; `subscribe_my_voice` debounces Deepgram finals through `VoiceLullMonitor` and then hands the merged utterance to the **same** monitor, keyed by voice channel id. On a YES decision, the monitor's `on_respond` callback dispatches to `_run_text_response` or `_run_voice_response` based on whether a voice handler is registered for the channel.

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

An earlier sketch suggested a "chattiness preset" governing response frequency alone. That misses the character layer: a "moderate" familiar should feel very different depending on who it is. A shy familiar hedges when speaking unprompted; an arrogant one acts like it's doing the room a favour; a curious one opens with a question. The preset governs *when* it responds; the personality description governs *how it feels* about responding. This design keeps both: `chattiness` for personality, `interjection` for mechanical timing.

## Configuration

All three settings live in `character.toml` on `CharacterConfig`:

```toml
aliases = ["aria", "ari"]
chattiness = "Curious and opinionated, but knows when to let others have their moment"
interjection = "average"
text_lull_timeout = 10.0
voice_lull_timeout = 5.0
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

Silence is detected by a client-side wall-clock timer in `VoiceLullMonitor` (`src/familiar_connect/voice_lull.py`). The Deepgram stream runs with `interim_results=false` + `endpointing=300` ms, so each `Results(is_final=true)` from Deepgram is a complete endpointed segment. Every Deepgram event — either a VAD pulse (`SpeechStarted`) or a final — re-arms the `voice_lull_timeout` timer; when the channel has been quiet for that long, the buffered finals merge and are handed to `ConversationMonitor.on_message` as one utterance keyed by the voice channel id and flagged with `is_lull_endpoint=True`. The lull is intentionally decoupled from both Deepgram `UtteranceEnd` (unreliable — Deepgram often holds the event for seconds under continuous audio) and from Discord audio-frame arrivals (client-side VAD varies; background noise keeps frames flowing through real silence).

For voice, **`voice_lull_timeout` is the only conversational pause** — the monitor treats each merged voice utterance as itself the endpoint of a lull, so the side-model YES/NO gate fires inline rather than waiting an additional `text_lull_timeout`. Direct address and counter-based interjection still fire on each merged utterance as they do for text.

**Default:** `5.0`

## Implementation

### `ConversationMonitor`

A standalone class (`src/familiar_connect/chattiness.py`) that owns all per-channel conversation state and is the single entry point `bot.py` calls instead of going straight to the pipeline. Held on the `Familiar` bundle, built in `Familiar.load_from_disk`.

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

**Timing:** Threshold starts at the tier's starting interval and decreases by 3 after each check, flooring at 3. Each interval has ±1–2 message jitter applied so the cadence is never exactly predictable.

**Extra prompt context:** `"{N} messages have been said without you speaking."` (where N is the total `message_counter` value, not the interval).

**Step-down curve.** The starting interval is tier-dependent. After each check where the familiar declines, the interval for the *next* check decreases by 3, with a floor of 3. The total message counter never resets until the familiar responds. Each computed interval has ±1–2 message jitter added before use (floor still 3), so the exact fire message varies slightly each cycle.

Approximate example for `average` (starting interval = 9, shown without jitter):

| Check # | Fires near message | Base interval | Next base |
|---------|-------------------|---------------|-----------|
| 1       | 9                 | 9             | 6         |
| 2       | 15                | 6             | 3         |
| 3       | 18                | 3             | 3 (floor) |
| 4       | 21                | 3             | 3         |
| ...     | every ~3 after    | 3             | 3         |

Approximate example for `very_quiet` (starting interval = 15):

| Check # | Fires near message | Base interval | Next base |
|---------|-------------------|---------------|-----------|
| 1       | 15                | 15            | 12        |
| 2       | 27                | 12            | 9         |
| 3       | 36                | 9             | 6         |
| 4       | 42                | 6             | 3         |
| 5       | 45                | 3             | 3 (floor) |
| ...     | every ~3 after    | 3             | 3         |

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

## Non-goals

- **Interruption of a reply already in progress.** This page covers the **pre-speech** decision (whether to start talking). Cutting off a reply mid-stream when someone else starts talking lives in [Voice interruption](interruption.md).
- **Silence-initiated conversation starts.** The lull trigger requires *some* messages in the buffer. A feature where prolonged silence with an empty buffer causes the familiar to proactively start a new conversation is architecturally different and deferred.
- **Per-modality evaluation prompts.** Text and voice now have separate lull timeouts (`text_lull_timeout` / `voice_lull_timeout`), but they still share the same evaluation prompt structure. Tuning prompts per modality is a future addition.

## Future work

- **Twitch events.** Twitch events (subs, bits, raids) should always be acknowledged. They could bypass the monitor entirely, or be treated as direct-address-equivalent triggers. Deferred until Twitch integration is wired into the monitor.
