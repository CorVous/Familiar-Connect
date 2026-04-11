# Voice Interruption Flow

!!! info "Status: Design — not committed"
    Ported from an earlier design PR. The mechanism is scoped
    specifically to voice channels as written, but latency and
    interruption handling is a spectrum that spans both voice and
    text modalities, so a modality-agnostic version of this design
    may be preferable. Revisit the scope before implementation — see
    [Voice input](voice-input.md) and
    [Conversation flow](conversation-flow.md) for the adjacent
    designs this one touches.

When a user speaks for at least `min_interruption_s` (configurable, default 1.5s) while the familiar is either generating a response or speaking, it counts as an **interruption**. This feature applies only to voice channels — text conversations are unaffected.

Back-channel sounds (brief "mm-hm", laughter, etc.) are naturally filtered out by the minimum duration threshold — no special audio classification is needed.

Multiple people speaking during the same interruption window are merged into a single interruption event.

---

## Configuration

### `character.toml`

```toml
[voice.interruption]
# Float 0.0–1.0. Probability of *continuing to talk* when interrupted
# while speaking. 0.0 = always yields (meek/polite),
# 1.0 = never yields (stubborn/dominant).
interrupt_tolerance = 0.3

# Minimum seconds of user speech before it counts as an interruption.
min_interruption_s = 1.5

# Seconds threshold separating "short" from "long" interruption.
short_long_boundary_s = 4.0
```

All three values live on `CharacterConfig` and are loaded at startup. They are not per-channel — they describe the familiar's personality, not the channel's tuning.

**Future (deferred):** Mood-based drift. A side-model evaluates the familiar's current emotional state and returns a modifier (e.g. `-0.2` to `+0.3`) added to `interrupt_tolerance` before each RNG roll. The base trait stays fixed in config; the modifier is ephemeral. Examples:
- Excited about a topic → tolerance drifts up (talks over people more)
- Just got corrected → tolerance drifts down (yields more readily)
- In a heated argument → tolerance spikes
- Asking a question → tolerance drops (wants the answer)

**Future (deferred):** When the familiar keeps talking through an interruption, the interrupted person's words could trigger a follow-up response after the current voice line finishes.

**Future (deferred):** Pre-generated interrupt tolerance. The RNG toll check could be replaced or augmented by a side-model evaluation that runs **before** playback begins — at generation time, the model decides whether the familiar would yield if interrupted for this particular response (e.g. a passionate rant vs. a throwaway comment). The result is cached on the `ResponseTracker` so that when an interruption actually arrives, the yield/keep-talking decision is instant with no latency cost. This becomes worthwhile if the decision logic grows beyond a single RNG roll (e.g. factoring in who is interrupting, emotional state, topic importance).

---

## Response State Machine

A per-guild `ResponseTracker` tracks the familiar's current phase:

```
IDLE ──→ GENERATING ──→ SPEAKING ──→ IDLE
              │              │
         (interrupted)  (interrupted)
```

Today there is no tracking — `_handle_voice_result` is fire-and-forget. The state machine is the prerequisite for everything below.

### `ResponseState` enum

| State | Meaning |
|---|---|
| `IDLE` | Not generating or speaking. Interruption detection is off. |
| `GENERATING` | LLM call is in-flight. TTS has not started. |
| `SPEAKING` | TTS audio is playing via `vc.play()`. |

The tracker also holds:
- The current `asyncio.Task` for the generation (so it can be cancelled)
- The full response text (once generation completes)
- The **word timestamp map** from Cartesia (for resume-from-word during SPEAKING)
- The **elapsed playback time** at the moment of interruption
- A reference to the voice client (`vc`)

---

## Interruption Detection

### Signal source

The voice pipeline already runs per-user Deepgram streams with VAD. Detection watches for **any user's speech activity** while the state is `GENERATING` or `SPEAKING`.

Relevant Deepgram events:
- `SpeechStarted` — user began talking (~200ms latency). Starts the interruption timer.
- Interim/final transcription — confirms real speech, captures what was said.
- `UtteranceEnd` — the interruption ended. Captures the total duration.

### Interruption lifecycle

1. **Detect start**: `SpeechStarted` fires while state ≠ `IDLE`. Record `interruption_start_time` and `interrupter_user_id`.
2. **Accumulate**: Buffer interim transcripts. If additional users speak during the same window, merge into the same event (single interruption, multiple contributors).
3. **Detect end**: `UtteranceEnd` or sustained silence (no `SpeechStarted` for ≥ `min_interruption_s`). Record `interruption_end_time`.
4. **Classify**: `duration = end - start`.
   - `duration < min_interruption_s` → **not an interruption** (back-channel). Discard.
   - `min_interruption_s ≤ duration < short_long_boundary_s` → **short**.
   - `duration ≥ short_long_boundary_s` → **long**.
5. **Dispatch** to the appropriate handler based on current state + classification.

---

## Scenario 1: Interrupted During Generation

The LLM call is in-flight. A user starts talking.

### Short interruption

- **Do not cancel** the generation — let it finish.
- **Hold** the completed response in a buffer.
- Gate delivery on an `asyncio.Event` that is set when silence resumes.
- Once silence is confirmed, proceed to TTS + playback as normal.
- Effect: the familiar pauses politely, then speaks as if nothing happened.

### Long interruption

- **Cancel** the in-flight LLM task (`generation_task.cancel()`).
- Capture the interruption transcript (what the user said).
- Build a **new `ContextRequest`** that includes:
  - The original utterance that triggered the generation.
  - The interruption content.
  - Interruption metadata injected as a system-level note:
    ```
    {speaker} interrupted while you were forming a response.
    They said: "{transcript}"
    ```
- Regenerate from scratch. The familiar's reply now accounts for the new context.

### Implementation notes

- `LLMClient.chat()` currently awaits the full HTTP response. Wrap the call in an `asyncio.Task` so it can be cancelled.
- On cancellation, the OpenRouter HTTP request should be aborted (`response.aclose()`) to avoid wasting tokens.
- The hold-and-deliver path needs no LLM changes — just an `asyncio.Event` gate between generation completion and TTS.

---

## Scenario 2: Interrupted While Speaking

TTS audio is playing via `vc.play()`. A user starts talking.

### Step 1: RNG toll check

```python
import random
keep_talking = random.random() < character_config.interrupt_tolerance
```

- If `keep_talking` is `True`: **continue playback**. The interrupted user's speech is still transcribed and appended to conversation history so nothing is lost. No further action for this interruption.
- If `keep_talking` is `False`: **stop playback** (`vc.stop()`). Proceed to step 2.

### Step 2: Classify and respond

#### Short interruption (stopped talking)

- Record the **word position** where playback was stopped, using Cartesia's word-level timestamps mapped against elapsed playback time.
- Once silence resumes, **re-synthesize and play the remaining text** from the cutoff word onward.
- No LLM re-generation needed — the familiar just picks up where it left off.
- The interruption transcript is still appended to history.

#### Long interruption (stopped talking)

- Record the word position and the partial text that was already spoken.
- Capture the interruption transcript.
- Build a new `ContextRequest` with enriched context:
  ```
  You were speaking and said: "{partial_text_delivered}"
  You were interrupted by {speaker}, who said: "{interruption_transcript}"
  Decide whether to finish your original thought, address what they
  said, or comment on being interrupted — whatever feels natural.
  ```
- The LLM decides the response. The familiar might:
  - Finish the original thought if still relevant.
  - Pivot to address the interruption.
  - React to being cut off (personality-driven).

---

## Word Position Tracking

Cartesia Sonic's streaming mode provides **word-level timestamps** in its response. We use these to map `elapsed_playback_time → word_index` when playback is stopped.

### Data flow

1. TTS synthesis returns audio bytes **plus** a list of `(word, start_time_ms, end_time_ms)` tuples.
2. These are stored on the `ResponseTracker` alongside the full response text.
3. On interruption during SPEAKING, compute:
   ```python
   elapsed_ms = (now - playback_start_time) * 1000
   last_spoken_idx = max(i for i, (_, _, end) in enumerate(timestamps) if end <= elapsed_ms)
   partial_delivered = " ".join(word for word, _, _ in timestamps[:last_spoken_idx + 1])
   remaining_text = " ".join(word for word, _, _ in timestamps[last_spoken_idx + 1:])
   ```
4. `partial_delivered` is used for the long-interruption LLM context.
5. `remaining_text` is used for the short-interruption resume path.

### Cartesia integration change

The current `CartesiaTTSClient.synthesize()` calls the bytes endpoint and returns raw PCM. To get word timestamps, switch to Cartesia's **streaming WebSocket** mode which emits `chunk` events with `word_timestamps`. The method signature changes:

```python
async def synthesize(self, text: str) -> TTSResult:
    """Returns audio bytes + word-level timestamps."""
```

```python
@dataclass
class WordTimestamp:
    word: str
    start_ms: float
    end_ms: float

@dataclass
class TTSResult:
    audio: bytes            # PCM mono
    timestamps: list[WordTimestamp]
```

---

## New Components

| Component | Location | Purpose |
|---|---|---|
| `ResponseState` enum | `voice/interruption.py` | IDLE / GENERATING / SPEAKING |
| `ResponseTracker` | `voice/interruption.py` | Per-guild state machine, holds generation task, response text, word timestamps, playback timing |
| `InterruptionDetector` | `voice/interruption.py` | Watches transcript queue during non-IDLE states, classifies short/long, dispatches handlers |
| `interrupt_tolerance` | `config.py` → `CharacterConfig` | Float trait on character config |
| `min_interruption_s` | `config.py` → `CharacterConfig` | Minimum speech duration to count as interruption |
| `short_long_boundary_s` | `config.py` → `CharacterConfig` | Threshold separating short from long |
| `TTSResult` / `WordTimestamp` | `tts.py` | Return type from Cartesia with audio + timestamps |
| Cancellable generation wrapper | `bot.py` | Wraps `LLMClient.chat()` in a cancellable `asyncio.Task` |
| Interruption context injection | `context/types.py` | New optional fields on `ContextRequest` for interruption metadata |

---

## Files Modified

| File | Change |
|---|---|
| `config.py` | Add `interrupt_tolerance`, `min_interruption_s`, `short_long_boundary_s` to `CharacterConfig`; parse from `[voice.interruption]` TOML section |
| `tts.py` | Switch Cartesia from bytes endpoint to streaming WebSocket; return `TTSResult` with word timestamps |
| `bot.py` | Wrap voice response generation in cancellable task; integrate `ResponseTracker`; replace `while vc.is_playing()` poll with state-machine-driven flow |
| `voice_pipeline.py` | Feed interruption-relevant events (SpeechStarted, UtteranceEnd) to `InterruptionDetector` when state ≠ IDLE |
| `context/types.py` | Add optional `interruption_context` field to `ContextRequest` |
| `context/render.py` | Render `interruption_context` as a system message when present |

---

## Sequence Diagrams

### Interrupted during generation (long)

```
User A speaks → Deepgram transcribes → pipeline fires → LLM generating...
                                                              │
User B speaks (>4s) ─── InterruptionDetector ─── long ───────►│
                                                              │
                                                    cancel generation task
                                                              │
User B stops speaking ─── capture transcript ─────────────────►│
                                                              │
                                              build new ContextRequest
                                              (original + interruption ctx)
                                                              │
                                                    new LLM generation
                                                              │
                                                         TTS + play
```

### Interrupted while speaking (short, yielded)

```
Familiar speaking ─── vc.play(audio) ───────────────────────────►
                                                                 │
User speaks (2s) ─── InterruptionDetector ─── RNG: yield ───────►│
                                                                 │
                                                     vc.stop()
                                                     record word position
                                                                 │
User stops speaking ─────────────────────────────────────────────►│
                                                                 │
                                               re-synth remaining text
                                                                 │
                                                     vc.play(remaining)
```

### Interrupted while speaking (long, yielded)

```
Familiar speaking ─── vc.play(audio) ───────────────────────────►
                                                                 │
User speaks (>4s) ─── InterruptionDetector ─── RNG: yield ─────►│
                                                                 │
                                                     vc.stop()
                                                     record partial text
                                                                 │
User stops speaking ─── capture transcript ─────────────────────►│
                                                                 │
                                              build ContextRequest
                                              (partial + interruption ctx)
                                                                 │
                                                    LLM generation
                                                                 │
                                                         TTS + play
```
