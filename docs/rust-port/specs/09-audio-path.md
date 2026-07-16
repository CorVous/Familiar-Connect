# 09-audio-path — port spec

Source files: `voice/{__init__,dave_client,dave_ws,recording_sink,audio}.py`,
`voice/turn_detection/{__init__,ten_vad,smart_turn,endpointer,factory}.py`,
`stt/{__init__,protocol,factory,deepgram,parakeet,faster_whisper}.py`,
`tts.py`, `tts_player/{__init__,protocol,discord_player,logging_player,mock}.py`,
`sentence_streamer.py` (~3,906 loc).

## Role

Everything audio: Discord voice in (DAVE-E2EE decrypt → per-SSRC PCM →
per-user turn detection + STT) and Discord voice out (LLM sentences → TTS
synthesis → PCM push through the voice client). It exposes three Protocol
seams the rest of the system types against — `Transcriber` (STT),
`TTSPlayer` (playback), and the duck-typed TTS-client surface
(`synthesize` / optional `synthesize_stream`) — plus the
`UtteranceEndpointer` state machine that decides when a spoken turn has
ended. The wiring layer (10) owns the per-user pump/demux tasks; this
subsystem defines every contract those tasks call into.

## Public API surface

### voice (DAVE E2EE client)

```python
class DaveVoiceClient(discord.VoiceClient):
    dave_session: davey.DaveSession | None      # None until negotiated
    dave_protocol_version: int                  # 0 = no DAVE
    _dave_pending_transitions: dict[int, int]   # transition_id -> protocol_version
    async def connect_websocket() -> DaveVoiceWebSocket
    async def _reinit_dave_session(ws=None) -> None   # called by WS layer on recovery
    def unpack_audio(data: bytes) -> None       # RX: SRTP-decrypted -> DAVE decrypt -> opus decode
    def _get_voice_packet(data: bytes) -> bytes # TX: opus -> DAVE encrypt -> RTP+SRTP

class DaveVoiceWebSocket(DiscordVoiceWebSocket):
    async def identify() -> None                             # adds max_dave_protocol_version
    async def poll_event() -> None                           # TEXT + BINARY dispatch (30 s recv timeout)
    async def received_message(msg) -> None                  # JSON ops 21/22/24 + SESSION_DESCRIPTION hook
    async def received_binary_message(op: int, data: bytes)  # ops 25/27/29/30
    async def send_dave_binary(op: int, payload: bytes)      # TX framing (no seq prefix)

class RecordingSink(discord.sinks.Sink):
    RecordingSink(*, loop, audio_queue: asyncio.Queue[tuple[int, bytes]], filters=None)
    write(data: bytes, user: int) -> None   # CALLED FROM PYCORD RECORD THREAD
    cleanup() -> None                       # sets finished=True
```

### voice.audio (conversion primitives)

```python
DISCORD_FRAME_SIZE = 3840   # 48000 Hz * 2 ch * 2 B * 0.020 s

class Resampler48to16:      # stateful streaming 3:1 boxcar decimator, int16
    feed(pcm_48k: bytes) -> bytes    # raises ValueError on odd length
    close() -> bytes                 # flush partial triplet zero-padded (<=1 sample out)
    reset() -> None                  # drop carry silently

def mono_to_stereo(data: bytes) -> bytes   # duplicate each s16le sample L+R; ValueError odd len
def stereo_to_mono(data: bytes) -> bytes   # (L+R)//2 per frame; ValueError len % 4 != 0

class StreamingPCMSource(discord.AudioSource):   # thread-safe producer/consumer PCM buffer
    StreamingPCMSource(*, prebuffer_bytes: int = 0, pad_underrun: bool = False)
    feed(data: bytes) -> None       # producer (asyncio task)
    close_input() -> None           # EOS flag; idempotent
    read() -> bytes                 # CALLED FROM PYCORD AUDIO THREAD; 3840-byte frames
    is_opus() -> bool               # always False
    cleanup() -> None               # == close_input(); pycord calls on player stop
```

### voice.turn_detection

```python
class TenVAD:                # per-user, stateful native handle (Agora ten-vad)
    TenVAD(*, sample_rate=16000, hop_size=256, threshold=0.5)
    speech_probability(pcm_chunk: bytes) -> float   # exactly hop_size int16 samples
    is_speech(pcm_chunk: bytes) -> bool             # prob >= threshold
    reset() -> None                                 # rebuild native handle

class SmartTurnDetector:     # shared, stateless ONNX classifier (Pipecat Smart Turn v3)
    SmartTurnDetector(model_path, *, sample_rate=16000, threshold=0.5, max_duration_s=16.0)
    completion_probability(pcm_audio: bytes) -> float
    is_complete(pcm_audio: bytes) -> bool

class UtteranceEndpointer:   # per-user state machine over 48 kHz mono s16le
    UtteranceEndpointer(*, vad, smart_turn, on_turn_complete: async (bytes) -> None,
                        silence_ms=200, speech_start_ms=100)
    async feed_audio(pcm_48k: bytes) -> None
    async force_complete_if_pending() -> bool    # idle-fallback entry point (pump calls it)
    reset() -> None

@dataclass LocalTurnDetector:   # per-process bundle; SmartTurn lazy-loaded + shared
    smart_turn_path: Path; silence_ms=200; speech_start_ms=100;
    smart_turn_threshold=0.5; vad_threshold=0.5; vad_hop_size=256; idle_fallback_s=1.5
    make_endpointer(*, on_turn_complete) -> UtteranceEndpointer   # fresh TenVAD per call

def create_local_turn_detector(config: LocalTurnConfig) -> LocalTurnDetector | None
    # hf_hub_download of ONNX weights; ANY failure -> None + warning (degrade, never raise)
```

### stt

```python
@dataclass TranscriptionResult:      # TranscriptionEvent is an alias of this type
    text: str; is_final: bool; start: float; end: float
    confidence: float = 0.0; speaker: int | None = None; user_id: int | None = None
    to_message(speaker_names: dict[int, str] | None) -> Message
        # role="user", content=f"[Voice] {text}", name: user_id match wins over speaker match, else "Voice"

@runtime_checkable class Transcriber(Protocol):        # <-- PROTOCOL SEAM
    clone() -> Transcriber                              # fresh per-user instance, same config
    async start(output: asyncio.Queue[TranscriptionEvent]) -> None
    async send_audio(data: bytes) -> None               # linear16, rate impl-defined (48 kHz here)
    async finalize() -> None                            # flush pending segment as final; no-op safe
    async stop() -> None                                # idempotent teardown

def create_transcriber(config: STTConfig) -> Transcriber       # ValueError on unknown backend /
    # missing secret / missing optional extra (lazy-import RuntimeError re-raised as ValueError)

class DeepgramTranscriber: ...        # streaming WS; also exposes endpointing_ms attr (bot pokes it)
def create_deepgram_transcriber(config: DeepgramSTTConfig | None) -> DeepgramTranscriber
class ParakeetTranscriber: ...        # buffer-and-finalize, NeMo; no endpointing_ms attr
class FasterWhisperTranscriber: ...   # buffer-and-finalize, CTranslate2; no endpointing_ms attr
DEFAULT_IDLE_FINALIZE_S = 0.5         # imported by bot.py for plain-Deepgram idle flush
```

### tts

```python
@dataclass(frozen) WordTimestamp: word: str; start_ms: float; end_ms: float
@dataclass(frozen) TTSResult: audio: bytes; timestamps: list[WordTimestamp] = []

class CartesiaTTSClient:
    CartesiaTTSClient(*, api_key, voice_id, model, base_url=..., ws_url=..., sample_rate=48000)
    async synthesize(text) -> TTSResult                    # one WS connection per call
    async synthesize_stream(text) -> AsyncIterator[bytes]  # raw mono pcm_s16le chunks

class AzureTTSClient:
    AzureTTSClient(*, subscription_key, region, voice_name="en-US-AmberNeural", sample_rate=48000)
    stream_prebuffer_bytes: int = 0      # duck-typed jitter hints read by DiscordVoicePlayer
    stream_pad_underrun: bool = False
    async synthesize(text) -> TTSResult                    # blocking SDK call in executor
    async synthesize_stream(text) -> AsyncIterator[bytes]  # worker thread + queue bridge

class GeminiTTSClient:
    GeminiTTSClient(*, api_key, voice_name, model, style_prompt=None, sample_rate=48000)
    async synthesize(text) -> TTSResult    # buffered only — NO synthesize_stream

def create_tts_client(tts_config: TTSConfig) -> Cartesia|Azure|Gemini   # ValueError on
    # unknown provider / missing env secret / missing cartesia voice_id or model
async def get_cached_greeting_audio(provider, voice_id, greeting, client) -> TTSResult
```

The TTS-client seam is duck-typed, not a declared Protocol:
`DiscordVoicePlayer` requires `synthesize`, takes the streaming path iff
`hasattr(client, "synthesize_stream")`, and reads optional
`stream_prebuffer_bytes` / `stream_pad_underrun` attrs via getattr-with-default.

### tts_player

```python
class TTSPlayer(Protocol):                                  # <-- PROTOCOL SEAM
    async speak(text: str, *, scope: TurnScope) -> None     # honour scope.is_cancelled()
    async stop() -> None                                    # flush in-flight audio now

class DiscordVoicePlayer:      # production impl
    DiscordVoicePlayer(*, tts_client, get_voice_client: () -> VoiceClientLike | None)
class LoggingTTSPlayer:        # no-audio default when no TTS client configured
    LoggingTTSPlayer(*, ms_per_word=200, poll_ms=20)
class MockTTSPlayer:           # test impl; records (text, was_cut) + total_played_ms,
    MockTTSPlayer(*, ms_per_word=200, poll_ms=5)   # exposes speak_started asyncio.Event
```

`DiscordVoicePlayer` types the voice client through a private structural
protocol (`is_connected/is_playing/play/stop`) so tests inject mocks — keep
the Rust equivalent a trait with exactly those four methods.

### sentence_streamer

```python
class SentenceStreamer:
    feed(delta: str) -> list[str]   # zero or more completed sentences per delta
    flush() -> str                  # drain partial buffer verbatim; resets state
```

## Behaviors & invariants

### A. DAVE E2EE (dave_client / dave_ws)

1. `identify()` sends the standard voice IDENTIFY payload plus
   `max_dave_protocol_version: davey.DAVE_PROTOCOL_VERSION`. Omitting it
   causes the gateway to close with 4017.
2. `connect_websocket()` polls until `secret_key` is set, then — only if
   `dave_protocol_version > 0` (read from SESSION_DESCRIPTION) — creates a
   fresh `DaveSession(version, bot_user_id, channel_id)` and sends the
   serialized MLS key package as binary op 26. The WS must be passed
   explicitly here because `self.ws` is not yet assigned by py-cord.
3. `poll_event()` dispatches BOTH text and binary WS frames (upstream
   py-cord silently drops binary). ERROR frame → raise ConnectionClosed;
   CLOSE/CLOSING/CLOSED → raise ConnectionClosed with `_close_code`. 30 s
   `wait_for` timeout on receive.
4. Binary RX frames update `seq_ack` from the 2-byte seq prefix BEFORE
   dispatch — if seq_ack doesn't track binary frames, gateway resume
   desyncs and the connection close-loops.
5. Binary TX frames are `[op: u8][payload]` — NO seq prefix. Adding one
   shifts the opcode and triggers close 4005.
6. SESSION_DESCRIPTION handling reads `d.dave_protocol_version` and stores
   it on the client BEFORE calling super(), so session init happens before
   the first binary MLS frame arrives. super() is always called so py-cord
   updates seq_ack for JSON frames.
7. Op 21 (PREPARE_TRANSITION): store `transition_id -> protocol_version` in
   pending map; if downgrading to version 0 and a session exists, call
   `set_passthrough_mode(True, 120)`; always reply op 23 TRANSITION_READY.
8. Op 22 (EXECUTE_TRANSITION): pop pending; unknown id → debug log + return
   (normal on initial handshake); known → set `dave_protocol_version`.
9. Op 24 (PREPARE_EPOCH): reinit session only when `epoch == 1` AND no
   session exists (a second MLS_KEY_PACKAGE after SESSION_DESCRIPTION init
   desyncs and closes the connection).
10. Op 25 (EXTERNAL_SENDER): `session.set_external_sender(payload)`; warn +
    drop if no session. Op 27 (PROPOSALS): first payload byte 0 → append,
    else revoke; `process_proposals` may return None (nothing to send) or a
    commit(+optional welcome); reply is op 28 with `commit || welcome`
    concatenated.
11. Ops 29 (ANNOUNCE_COMMIT_TRANSITION) / 30 (WELCOME): payload =
    `[transition_id: u16 BE][data]`. `process_commit`/`process_welcome`
    raising ValueError/RuntimeError → send op 31 INVALID_COMMIT_WELCOME
    with the transition_id, then reinit the whole DaveSession (fresh key
    package) — recovery, not crash. Success → reply op 23.
12. All binary handlers length-check their payload and warn+drop on short
    frames; a missing session is warn+drop, never a crash.
13. RX audio (`unpack_audio`): drop unless `data[1] & 0x78 == 0x78`; drop
    when paused; drop silence sentinel `b"\xf8\xff\xfe"`. When session
    exists AND `session.ready` AND ssrc→user_id known: DAVE-decrypt the
    SRTP-decrypted payload with `(user_id, MediaType.audio)`; ANY decrypt
    exception → drop the frame silently (debug log). Unknown SSRC or
    missing user_id → pass through undecrypted to the opus decoder.
14. TX audio (`_get_voice_packet`): layering is opus frame → DAVE
    `encrypt_opus` (only when session ready) → 12-byte RTP header
    (0x80, 0x78, seq u16 BE, timestamp u32 BE, ssrc u32 BE) → SRTP
    encrypt via the mode-selected `_encrypt_{mode}` method.

### B. Recording sink & PCM conversion

15. py-cord calls `RecordingSink.write(data, user)` from its recording
    thread. It must not await: it converts 48 kHz stereo s16le →
    mono ((L+R)//2, Python floor division) and hands `(user_id, mono)` to
    the asyncio queue via `loop.call_soon_threadsafe(put_nowait, ...)`.
    Queue is unbounded; backpressure is deliberate non-existence here.
16. `stereo_to_mono` rejects lengths not divisible by 4; `mono_to_stereo`
    rejects odd lengths; both raise ValueError with the length in the
    message. `mono_to_stereo` output must be byte-identical between the
    numpy fast path and the pure loop (pinned by test) — in Rust one
    implementation suffices but must match `<i2` little-endian layout.
17. CRITICAL parity trap: all sample averaging uses Python `//`
    (floor toward −∞). Rust integer `/` truncates toward 0. E.g.
    `(-3 + 0) // 2 == -2` in Python but `-3/2 == -1` in Rust. Applies to
    `stereo_to_mono`, `Resampler48to16` (sum of 3 // 3), and Gemini
    upsampling `(a+b)//2`. Use `i32::div_euclid` or match floor semantics
    explicitly; tests pin exact byte outputs.

### C. Resampler48to16

18. Streaming 3:1 decimation with boxcar prefilter: each output sample is
    the integer floor-mean of 3 consecutive int16 inputs. Carries 0–2
    leftover samples across `feed` calls (arbitrary chunk lengths in, no
    input alignment requirement beyond even byte count).
19. `close()` zero-pads a partial triplet and emits at most one sample;
    empty carry → `b""`. `reset()` drops carry without emitting. Aliasing
    above 8 kHz is accepted by design (TEN-VAD tolerates it).

### D. StreamingPCMSource (producer/consumer, cross-thread)

20. One `threading.Condition` guards buffer + closed + primed. `feed`
    appends and notifies one waiter; empty feed is a no-op; `close_input`
    sets EOS and notifies all.
21. `read()` (consumer thread) semantics, in order: (a) if not primed,
    block until `len(buf) >= prebuffer_bytes` OR closed, then latch primed
    (pre-roll gates ONLY the first read); (b) if short of one 3840-byte
    frame and not closed: return one frame of zeros when
    `pad_underrun=True`, else block until a full frame or EOS; (c) full
    frame available → pop exactly 3840 bytes; (d) EOS with partial
    remainder → zero-pad to 3840 and return it (final frame plays); (e)
    EOS and empty → return `b""` (signals pycord to stop the player).
22. EOS always overrides both pre-roll and underrun padding (a short reply
    must play out; padding must not loop forever). `cleanup()` (called by
    pycord on player stop) == `close_input()`, releasing any blocked
    reader — without this a cancelled playback deadlocks the audio thread.

### E. Turn detection

23. `TenVAD`: constructor rejects `sample_rate != 16000` and hop_size not
    in {160, 256} (ValueError), and raises RuntimeError mentioning the
    `local-turn` extra when the native package is absent.
    `speech_probability` requires exactly hop_size samples (ValueError
    otherwise); thresholding is done in the wrapper from the returned
    probability (the native flag is ignored) so threshold can be re-tuned
    without rebuilding the handle. `reset()` rebuilds the native handle —
    the only way to clear its accumulated state.
24. `SmartTurnDetector`: input = int16 bytes → float32 / 32768.0; if
    longer than `max_duration_s * 16000` samples keep the most recent
    window (tail — turn-end semantics live there). Input tensor shape
    `[1, N]`, fed under the model's first graph input name (Pipecat uses
    `input_values`). Output handling: last dim 2 → numerically-stable
    softmax, take class 1; last dim 1 → sigmoid; anything else →
    ValueError. Stateless and shared across users.
25. `UtteranceEndpointer` framing: 48 kHz feed → resampler → byte carry →
    consume exact 512-byte (256-sample, 16 ms) VAD frames. Chunk
    thresholds: `silence_chunks = max(1, int(silence_ms / 16.0))`,
    `speech_chunks = max(1, int(speech_start_ms / 16.0))` — note int()
    truncation: defaults 200 ms → 12 frames = 192 ms effective, 100 ms →
    6 frames = 96 ms. Preserve this rounding.
26. Per-frame step (`_on_vad_frame`): the frame is ALWAYS appended to the
    utterance buffer BEFORE the VAD verdict (so pre-latch ramp audio is
    included). Speech frame: silence_streak=0; while not SPEAKING,
    speech_streak++ and latch SPEAKING (clearing POST_INCOMPLETE) at
    threshold. Silence frame: speech_streak=0; if not SPEAKING → clear the
    utterance buffer (idle memory bound — this ALSO runs in
    POST_INCOMPLETE, draining the held audio while the state stays
    stranded) and return; else silence_streak++ until threshold.
27. At the silence-after-speech edge: skip classification entirely if
    POST_INCOMPLETE (extra silence never re-fires the classifier; only a
    fresh speech burst followed by a fresh silence streak does). Otherwise
    run `smart_turn.is_complete(buffer_copy)` via `asyncio.to_thread` —
    off-loop because wav2vec2 over up to 16 s of audio can stall Deepgram
    keepalives and the 10 s Discord voice heartbeat. Verdict complete →
    snapshot buffer, clear all state, `vad.reset()`, `await
    on_turn_complete(audio)`. Verdict incomplete → POST_INCOMPLETE
    (speaking=False), buffer retained (until the next not-speaking silence
    frame clears it, see 26).
28. `force_complete_if_pending()`: fires on STATE (`SPEAKING` or
    `POST_INCOMPLETE`), NOT on buffered bytes — the payload may legally be
    empty because of the idle-clear in 26; consumers key off the turn
    ending (STT finalize), not the audio. Emits buffer, resets everything
    incl. `vad.reset()`, awaits callback, returns True; returns False when
    idle (and after a normal completion — must not double-fire). Called by
    the 10-layer pump after `idle_fallback_s` of queue silence because
    Discord client VAD halts RTP and the frame-driven machine can never
    time out on its own.
29. `LocalTurnDetector.make_endpointer` builds a fresh TenVAD per user
    (stateful native handle) and lazily loads a single shared
    SmartTurnDetector on first call. `create_local_turn_detector` returns
    None — never raises — on missing huggingface_hub, download failure, or
    cache-rot (path returned but missing on disk), each with a distinct
    warning reason; the bot then falls back to Deepgram-only endpointing.

### F. STT

30. `Transcriber` lifecycle (all backends): `clone()` per user →
    `start(queue)` once → `send_audio(pcm)` per chunk → `finalize()` to
    flush → `stop()` idempotent. Feed rate is 48 kHz mono s16le from the
    sink. Results are pushed to the caller-provided asyncio queue in wire
    order; the wiring layer stamps `user_id` on each result in its fan-in
    task (results leave the backend with `user_id=None`).
31. `create_transcriber` known backends: `{"deepgram", "parakeet",
    "faster_whisper"}`; unknown → ValueError naming the config key.
    Parakeet/FasterWhisper are lazy-imported; their import-time
    RuntimeError (numpy missing) is re-raised as ValueError so the run
    command degrades uniformly (logs a warning, continues text-only).

#### DeepgramTranscriber

32. Constructor defaults: model `nova-3`, language `en`, sample_rate
    48000, channels 1, diarize False, interim_results True,
    utterance_end_ms 1500, vad_events False, endpointing_ms 500,
    smart_format True, punctuate True, keyterms (), replay_buffer_s 5.0.
    Class-attr tunables (copied by `clone()` and set by the factory):
    `_KEEPALIVE_INTERVAL=3.0`, `_MAX_RECONNECTS=5`,
    `_RECONNECT_DELAY=1.0`, `_RECONNECT_BACKOFF_CAP=16.0`,
    `_IDLE_CLOSE_S=30.0` (read by the 10-layer to arm its per-user idle
    watchdog; 0 disables).
33. `start(queue)` opens the WS and spawns exactly two tasks: receive loop
    and keepalive loop. `send_audio` before `start` raises RuntimeError;
    `finalize` before `start` is a silent no-op (idle-watchdog safe).
34. `send_audio` ordering: under `_send_lock`, append to replay buffer
    FIRST (so a mid-send drop is still buffered), then skip the actual
    send when `ws.closed` or `_closing`, else send bytes suppressing all
    exceptions. Replay buffer is a deque of whole chunks with byte
    accounting; eviction pops oldest whole chunks while total >
    `replay_buffer_s * sample_rate * channels * 2` bytes.
35. `_closing` flag: set the moment the receive loop OBSERVES the server
    CLOSE frame (this is why the loop uses explicit `receive()` rather
    than `async for`, which swallows CLOSE) so writers (audio +
    keepalive) stop racing the closing transport — writing in that window
    raises ClientConnectionResetError and corrupts a clean 1000 close into
    a 1006. Cleared only after a successful reconnect.
36. Receive loop: TEXT frames parse as JSON; `type=="Results"` →
    `_parse_response` → queue put, and any real result resets the
    consecutive-reconnect counter to 0. Non-Results types are logged with
    the FULL payload (Metadata carries session-end diagnostics).
    CLOSE → set `_closing`, capture reason, break; CLOSING/CLOSED/ERROR →
    break.
37. `_parse_response` returns None unless: type is "Results", alternatives
    non-empty, and `alternatives[0].transcript` non-empty. Fields:
    `is_final=bool(data.is_final)`, `start=data.start`,
    `end=start+data.duration`, `confidence=alternatives[0].confidence`,
    `speaker` = int of `words[0].speaker` when present (diarization).
38. Reconnect policy, after loop exit: if `_shutting_down` (set by
    `stop()` BEFORE sending CloseStream) → exit silently. Else classify
    close code: non-int (transport drop) → retry; 1008 → give up; 4000 ≤
    code < 5000 (Deepgram app errors: auth/billing/quota) → give up; else
    retry. Retries bounded by `_MAX_RECONNECTS` consecutive failures;
    backoff: attempt 1 immediate, attempt n≥2 waits
    `min(1.0 * 2^(n-2), cap)` seconds. Any exception during reconnect
    itself → log + terminate loop permanently.
39. `_reconnect()`: close old ws+session, open fresh ones, clear
    `_closing`, then UNDER `_send_lock` replay every buffered chunk (so
    live `send_audio` callers queue behind the drain rather than
    interleaving) and clear the buffer. If anything was replayed: sleep
    `replay_bytes/(rate*ch*2) + DEFAULT_IDLE_FINALIZE_S` seconds OUTSIDE
    the lock (replay arrives faster than real-time; immediate Finalize
    would flush a partial), then send `{"type":"Finalize"}` under the
    lock. Empty replay sends no Finalize.
40. Keepalive loop: every `_KEEPALIVE_INTERVAL` s, re-read `self._ws`
    (follows reconnects), skip when None/closed/`_closing`, send
    `{"type":"KeepAlive"}` suppressing errors; survives send failures
    forever; cancelled by `stop()`.
41. `stop()` order matters: set `_shutting_down` → cancel+await keepalive
    → send `{"type":"CloseStream"}` + close ws (each suppressed) → null ws
    → cancel+await receive task → close session. Idempotent.
42. `clone()` copies the full constructor config AND the four class-attr
    tunables onto the new instance; the clone owns an independent WS.

#### Parakeet / FasterWhisper (buffer-and-finalize backends)

43. `send_audio`: 48 kHz PCM → shared `Resampler48to16` → append 16 kHz
    bytes to an instance buffer. No output until `finalize()`. These
    backends REQUIRE the local turn detector (or idle fallback) to drive
    `finalize`; they never emit on their own.
44. `finalize()` under a per-instance asyncio lock: no-op when buffer
    empty OR output queue unset OR model unloaded; otherwise convert
    int16 → float32/32768, compute `duration = samples/16000`, CLEAR THE
    BUFFER BEFORE inference (so an overlapping finalize from the idle
    watchdog no-ops instead of double-transcribing), run inference via
    `asyncio.to_thread`, drop empty text, else emit exactly one
    `TranscriptionResult(text, is_final=True, start=0.0, end=duration)`.
45. `clone()` shares the loaded model handle (load once per process) and
    `_IDLE_CLOSE_S`; `start()` lazy-loads the model in a thread on first
    call only. `stop()` clears buffer + resets resampler + drops the
    output queue reference but keeps the model.
46. Parakeet must handle both NeMo return shapes (list[str] and
    list[Hypothesis.text]); FasterWhisper must CONSUME the segments
    generator (lazy — unconsumed means no inference) and join
    `seg.text` then strip. Both deliberately DO NOT expose an
    `endpointing_ms` attribute — the 10-layer's
    `hasattr(clone, "endpointing_ms")` guard keys off this (pinned).

### G. TTS clients

47. Cartesia `synthesize`: fresh ClientSession + WS per call; auth in the
    query string (`api_key`, `cartesia_version=2024-06-10`); send the
    payload (see Data formats), then consume TEXT events: `chunk` →
    base64-decode and accumulate; `timestamps` → parse parallel arrays
    (seconds → ms); `done` → finish; `error` → RuntimeError with
    status_code+message; CLOSED/ERROR WS frame → RuntimeError "closed
    unexpectedly". WS always closed in finally.
48. Cartesia `synthesize_stream`: same wire flow but yields each decoded
    chunk immediately, SKIPS empty chunks, and silently drops
    `timestamps` events. Errors raise identically. This is an async
    generator — early `aclose()` by the consumer must close the WS
    (finally block).
49. Azure `synthesize`: whole blocking SDK call runs in the default
    executor. Output format `Raw48Khz16BitMonoPcm`; `audio_config=None`
    (audio comes back in `result.audio_data`). Word-boundary events are
    collected on the SDK's callback thread (no locking needed — same
    thread as `.get()`): only `BoundaryType.Word` counts;
    `start_ms = audio_offset / 10_000` (100 ns ticks),
    `end_ms = start_ms + duration.total_seconds()*1000`. Non-completed
    result reason → RuntimeError from cancellation details.
50. Azure `synthesize_stream`: worker thread runs
    `start_speaking_text_async` + `AudioDataStream.read_data` into a
    reused 32 KiB buffer, bridging each chunk to the async generator via
    `loop.call_soon_threadsafe(queue.put_nowait, ...)`; sentinel None =
    EOF, a BaseException instance = re-raise in the consumer. THE BUFFER
    MUST BE COPIED PER READ: `read_data` fills the same `bytes` object in
    place via ctypes, and a whole-slice `buffer[:n]` when `n == len` is
    the SAME object — the copy must be a fresh allocation
    (`bytes(memoryview(buffer)[:n])`).
51. Azure stream termination: `read_data == 0` is ambiguous (EOF vs
    mid-stream failure). Only when the stop event was NOT set and
    `stream.status == Canceled` raise from `cancellation_details`.
    Cancellation at start (`result.reason == Canceled`) raises
    immediately through the bridge.
52. Azure early close (barge-in): generator `finally` sets the stop
    event, then in an executor calls `stop_speaking_async().get()`
    (unblocks the in-flight read) and joins the worker with a 2.0 s
    bound; the worker is a daemon so a wedged SDK read can never pin
    shutdown.
53. Azure exposes `stream_prebuffer_bytes = 0` / `stream_pad_underrun =
    False` as class attrs — currently the plain path (delivery measured
    faster-than-realtime) but the knobs are the documented reversal
    lever; the player must keep reading them dynamically.
54. Gemini `synthesize` (blocking, in executor): contents =
    `f"{style_prompt}\n\n{text}"` when a style prompt is set, else the
    bare text; response modality AUDIO with prebuilt voice config. A
    missing audio part anywhere along
    candidates[0].content.parts[0].inline_data.data → RuntimeError
    "Gemini TTS returned no audio part". Native 24 kHz mono s16le output
    is 2x-upsampled by linear interpolation: each pair (a,b) emits
    (a, (a+b)//2); the final sample is doubled. Word timestamps are
    ESTIMATED: total_ms spread uniformly across whitespace-split words of
    the ORIGINAL text (not the style prompt); empty text or zero
    duration → empty list.
55. `_compose_gemini_style_prompt` emits, in order and only for set
    fields: `Audio Profile: {profile}`, `Scene: {scene} {context}`
    (space-joined, either alone works), `Director's Notes: Style: X.
    Pace: Y. Accent: Z.` — newline-joined, suffixed `"\nSay:"`; all six
    fields unset → None.
56. Greeting cache: path
    `data/cache/greetings/{sha256(f"{provider}:{voice_id}:{greeting}")}.bin`,
    file contains raw audio bytes only. Hit → return audio with EMPTY
    timestamps; miss → `client.synthesize`, write bytes, return audio
    with EMPTY timestamps (timestamps are never cached or returned). All
    file I/O off-loop.
57. `create_tts_client` env secrets: azure → AZURE_SPEECH_KEY +
    AZURE_SPEECH_REGION (each missing raises its own ValueError);
    cartesia → CARTESIA_API_KEY plus non-empty `cartesia_voice_id` and
    `cartesia_model` from config; gemini → GOOGLE_API_KEY, falling back to
    GEMINI_API_KEY (GOOGLE wins when both set).

### H. DiscordVoicePlayer

58. `speak` preconditions, in order: cancelled scope → return;
    empty/whitespace text → warn + return (Cartesia 400s on empty
    transcript); then streaming path iff the client has
    `synthesize_stream`, else buffered.
59. A single `asyncio.Lock` (`_play_lock`) serializes ALL playback across
    speakers: turn scopes are per-user (06), but the voice client is
    single-track — a second concurrent `vc.play` raises
    `ClientException('Already playing audio.')`. The lock makes speak #2
    wait for speak #1's playback to finish. FIFO fairness of the tokio
    equivalent matters for reply ordering.
60. Streaming path (lock held for synthesis AND playback): build
    `StreamingPCMSource` with the client's duck-typed jitter attrs;
    re-check cancel; get voice client (None or disconnected → warn +
    return); open the stream and await the FIRST chunk inline —
    StopAsyncIteration → warn "empty_stream" + return without playing;
    exception → warn + return; scope cancelled after first chunk →
    `aclose()` the generator + return. Then feed
    `mono_to_stereo(first_chunk)`, spawn the drain task (named
    `tts-stream-{turn_id}`), call `vc.play(source)` (ClientException →
    close input, cancel + await drain, return), record budget phase
    `playback_start` AFTER play, then poll `vc.is_playing()` every 20 ms;
    on cancel → `vc.stop()` + bounded stop-drain + return. Finally-block:
    `close_input()` + await the drain task (suppressing errors) so the
    producer always unwinds.
61. Drain task: per chunk, stop early on scope cancel; feed
    `mono_to_stereo(chunk)`; any exception → warn (never propagate);
    ALWAYS `close_input()` in finally (lets the reader hit EOF and pycord
    stop the player cleanly).
62. Buffered path: `synthesize` OUTSIDE the lock (exception → warn +
    return; cancel after synth → return; no/disconnected vc → warn +
    return), then under the lock re-check cancel, `mono_to_stereo`, wrap
    in `discord.PCMAudio(BytesIO)`, `vc.play`, record `playback_start`,
    same 20 ms cancel-poll loop with `vc.stop()` + stop-drain.
63. Stop-drain: after `vc.stop()`, poll `is_playing()` every 20 ms up to
    200 ms before releasing the lock. Pycord's audio thread only observes
    the stop flag once per 20 ms tick; releasing the lock immediately
    lets the next `speak` race into `ClientException`. `is_playing()`
    raising → treat as stopped. (Pinned by
    `test_cancel_then_immediate_speak_does_not_collide`.)
64. `stop()` (the Protocol method): fetch vc; None → return;
    `is_playing()` raising → return; playing → `vc.stop()`. It does NOT
    take the play lock — it's the barge-in fast path.
65. Budget telemetry (01): exactly one `playback_start` stamp per speak,
    recorded immediately after a successful `vc.play`. The recorder is
    best-effort/swallowing; the player never awaits it.
66. Logging/Mock players: pace at `ms_per_word` per whitespace word,
    polling cancel/stop every `poll_ms` via `wait_for(stop_event.wait(),
    step)`; the stop event is re-created per `speak` call (a prior stop
    must not kill the next utterance). Mock additionally records
    `(text, was_cut)` per call, accumulates `total_played_ms`, and sets
    `speak_started` when entered (barge-in tests await it).

### I. SentenceStreamer

67. Terminators are `.` `!` `?`. A boundary requires: terminator run
    (consecutive terminators collapse — `?!` is one boundary) followed by
    at least one char that `isspace()` (newline counts). Terminator at
    buffer end → wait for the next delta (never emit on a trailing dot).
    Non-space follower (`.5`, `1.0`, `?<tag>`) → not a boundary, scan on.
68. Abbreviation guard applies only when the run contains a `.`: walk
    back over `[a-zA-Z.]` to extract the preceding token (inner dots kept
    so `e.g` / `i.e` round-trip), lowercase it; suppress the boundary if
    it's in the abbreviation set {mr, mrs, ms, dr, st, sr, jr, prof, rev,
    fr, etc, vs, no, vol, pg, ft, e.g, i.e} or is a single alphabetic
    letter (initials: `J. K. Rowling`). Empty token → boundary stands.
69. On split: the emitted sentence INCLUDES the terminator run; ALL
    following whitespace is consumed (not carried into the remainder).
    `feed` loops so one delta can emit multiple sentences. `flush`
    returns the remaining buffer verbatim (untrimmed) and resets;
    `<silent>` (no terminator-then-space) never emits via feed — it
    reaches the 06 gate via flush.

### J. Contracts with the 10-layer pump (context, not owned here)

70. The wiring builds ONE transcriber clone + (optionally) ONE endpointer
    per Discord user_id, lazily on first audio. When a local detector is
    active AND the clone `hasattr(clone, "endpointing_ms")`, the wiring
    sets `clone.endpointing_ms = 10` BEFORE `start()` so Deepgram's
    hosted endpointer effectively never fires and finals are
    Finalize-driven. This attribute poke is part of Deepgram's public
    contract in the Rust port (a setter or builder option).
71. The endpointer's `on_turn_complete` callback (built by the wiring)
    first parks a `vad_end` perf-counter on `VoiceSource.record_vad_end
    (user_id)` and then calls `transcriber.finalize()` (exceptions
    suppressed). Callback exceptions must therefore never corrupt
    endpointer state — `feed_audio` is awaited from the same pump task
    that also awaits `send_audio`, in FIFO per user: `send_audio(pcm)`
    then `feed_audio(pcm)` for every chunk.
72. Idle-flush fallback timings the port must preserve: plain Deepgram →
    `finalize()` after `DEFAULT_IDLE_FINALIZE_S` (0.5 s) of per-user
    queue silence, but ONLY while "dirty" (audio sent since last flush);
    local detection → `force_complete_if_pending()` after
    `idle_fallback_s` (1.5 s default). A clean (non-dirty) queue must
    block indefinitely, not re-finalize every window.
73. The per-user idle watchdog closes a user's clone (+ endpointer +
    pump) after `_IDLE_CLOSE_S` (default 30 s, scan interval =
    max(idle_close_s/4, 0.01)) of no audio; everything reopens lazily on
    the next chunk. Teardown cancels pump + fan-in tasks before awaiting
    `clone.stop()`, and channel teardown stops all clones in PARALLEL
    (gather) to fit Discord's 3 s interaction deadline.

## Data formats

### PCM framing (canonical rates)

| Point | Format |
|---|---|
| Discord RX (post-opus-decode, sink input) | 48 kHz s16le stereo |
| Sink output / transcriber + endpointer input | 48 kHz s16le mono ((L+R)//2) |
| VAD / SmartTurn / local STT input | 16 kHz s16le mono (3:1 boxcar decimation) |
| TEN-VAD frame | 256 samples = 512 bytes = 16 ms (alt hop 160 = 10 ms) |
| SmartTurn input tensor | float32 [1, N], int16/32768.0, N ≤ 256 000 (16 s tail) |
| TTS client output (all three) | 48 kHz s16le MONO (Gemini upsampled 24 k→48 k) |
| Discord TX (vc.play source) | 48 kHz s16le STEREO, 20 ms frames of 3840 bytes |

### DAVE voice-gateway wire

- IDENTIFY (JSON op 0 `d`): `server_id`(str), `user_id`(str), `session_id`,
  `token`, `max_dave_protocol_version`(int).
- JSON opcodes: 21 PREPARE_TRANSITION `{transition_id, protocol_version}`,
  22 EXECUTE_TRANSITION `{transition_id}`, 23 TRANSITION_READY (TX)
  `{transition_id}`, 24 PREPARE_EPOCH `{epoch, protocol_version}`,
  31 INVALID_COMMIT_WELCOME (TX) `{transition_id}`.
- Binary RX frame: `[seq: u16 BE][op: u8][payload]`. Binary TX frame:
  `[op: u8][payload]` (no seq!). Binary opcodes: 25 EXTERNAL_SENDER,
  26 KEY_PACKAGE (TX), 27 PROPOSALS (`[optype u8: 0=append else revoke]
  [proposals]`), 28 COMMIT_WELCOME (TX, `commit || welcome?`),
  29 ANNOUNCE_COMMIT_TRANSITION (`[transition_id u16 BE][commit]`),
  30 WELCOME (`[transition_id u16 BE][welcome]`).
- RTP header (TX): 12 bytes `80 78 | seq u16 BE | timestamp u32 BE | ssrc
  u32 BE`. RX filter: `data[1] & 0x78 == 0x78`; silence sentinel payload
  `f8 ff fe`.

### Deepgram `/v1/listen` WS

- URL: `wss://api.deepgram.com/v1/listen?` + urlencoded params in this
  order: `model`, `language`, `sample_rate`, `channels`,
  `encoding=linear16`, `vad_events`, `endpointing` (ms int),
  `smart_format`, `punctuate`; then when interims on: `interim_results=
  true`, `utterance_end_ms`; then `diarize=true` when set; then one
  `keyterm=<term>` pair PER term. Booleans lowercased.
- Header: `Authorization: Token <DEEPGRAM_API_KEY>`.
- Client control messages (JSON text): `{"type":"KeepAlive"}`,
  `{"type":"Finalize"}`, `{"type":"CloseStream"}`. Audio = binary frames
  of raw linear16.
- Server `Results` shape consumed: `{type:"Results", start: f64,
  duration: f64, is_final: bool, channel:{alternatives:[{transcript:str,
  confidence:f64, words:[{speaker?:int,...}]}]}}`.
- Close codes: 1008 and 4000–4999 are terminal; everything else (incl.
  missing code) reconnects.

### Cartesia TTS WS

- URL: `wss://api.cartesia.ai/tts/websocket?api_key=<key>&
  cartesia_version=2024-06-10`. (REST headers, used by tests:
  `X-API-Key`, `Cartesia-Version`, `Content-Type: application/json`.)
- Request (JSON text): `{context_id: uuid4-hex, model_id, transcript,
  voice:{mode:"id", id}, output_format:{container:"raw",
  encoding:"pcm_s16le", sample_rate:48000}, language:"en",
  add_timestamps:true, continue:false}`.
- Events (JSON text): `{type:"chunk", data:<base64 pcm>}`,
  `{type:"timestamps", word_timestamps:{words:[str], start:[f64 s],
  end:[f64 s]}}` (parallel arrays, zip to min length, seconds→ms),
  `{type:"done"}`, `{type:"error", error, status_code}`.

### Azure / Gemini

- Azure output format enum: `Raw48Khz16BitMonoPcm`; word-boundary
  offsets in 100 ns ticks (÷10 000 → ms), durations as timedelta.
- Gemini: `generate_content` with `response_modalities=["AUDIO"]`,
  prebuilt voice config; returns 24 kHz s16le mono in
  `candidates[0].content.parts[0].inline_data.data`.

### Greeting cache (on disk)

`data/cache/greetings/<hex sha256 of "provider:voice_id:greeting">.bin` —
raw PCM bytes exactly as returned by `synthesize`, no header, no
timestamps.

### Voice-budget phases (emitted into 01)

This subsystem stamps `playback_start` (DiscordVoicePlayer, both paths).
Adjacent phases `vad_end` (parked via `VoiceSource.record_vad_end` from the
endpointer callback) and `stt_final` are stamped in `sources/voice.py`.
Turn keys are the bus `turn_id` strings.

## Config knobs

### `[providers.stt]` (STTConfig)

| Key | Default |
|---|---|
| `backend` | `"deepgram"` (also: `parakeet`, `faster_whisper`) |

`[providers.stt.deepgram]` (DeepgramSTTConfig): `model="nova-3"`,
`language="en"`, `endpointing_ms=500`, `utterance_end_ms=1500`,
`smart_format=true`, `punctuate=true`, `keyterms=[]`,
`replay_buffer_s=5.0`, `keepalive_interval_s=3.0`,
`reconnect_max_attempts=5`, `reconnect_backoff_cap_s=16.0`,
`idle_close_s=30.0`.

`[providers.stt.parakeet]`: `model_name="nvidia/parakeet-tdt-0.6b-v3"`,
`device="auto"`, `idle_close_s=30.0`.

`[providers.stt.faster_whisper]`: `model_size="small"`, `device="auto"`,
`compute_type="auto"`, `language="en"`, `idle_close_s=30.0`.

### `[providers.turn_detection]` (TurnDetectionConfig)

`strategy = "deepgram"` (default) | `"ten+smart_turn"`.

`[providers.turn_detection.local]` (LocalTurnConfig):
`smart_turn_repo_id="pipecat-ai/smart-turn-v3"`,
`smart_turn_filename="smart-turn-v3.2-cpu.onnx"`, `silence_ms=200`,
`speech_start_ms=100`, `vad_threshold=0.5`, `smart_turn_threshold=0.5`,
`vad_hop_size=256`, `idle_fallback_s=1.5`.

### `[tts]` (TTSConfig)

`provider="azure"` | `cartesia` | `gemini`; `cartesia_voice_id=None`,
`cartesia_model=None` (both REQUIRED for cartesia),
`azure_voice="en-US-AmberNeural"`, `gemini_voice="Kore"`,
`gemini_model="gemini-3.1-flash-tts-preview"`, six optional Gemini style
fields (`gemini_scene/context/audio_profile/style/pace/accent`),
`greetings=[]`.

### Environment variables

| Var | Used by |
|---|---|
| `DEEPGRAM_API_KEY` | deepgram factory (required for backend) |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | azure TTS factory |
| `CARTESIA_API_KEY` | cartesia TTS factory |
| `GOOGLE_API_KEY` (pref) / `GEMINI_API_KEY` | gemini TTS factory |
| `HF_HUB_OFFLINE=1` | (indirect) forces smart-turn weights cache-only |

Hard-coded constants worth surfacing as Rust consts:
`DEEPGRAM_WS_URL`, `CARTESIA_WS_URL/BASE_URL/API_VERSION`,
`DEFAULT_SAMPLE_RATE=48000`, `GEMINI_SAMPLE_RATE=24000`,
`DEFAULT_IDLE_FINALIZE_S=0.5`, `_POLL_S=0.02`, `_STOP_DRAIN_S=0.2`,
`_AZURE_STREAM_BUFFER_BYTES=32768`, `_AZURE_STREAM_JOIN_TIMEOUT_S=2.0`,
`_AZURE_TICKS_PER_MS=10_000`, `_GREETING_CACHE_DIR=data/cache/greetings`.

## Dependency edges

Imports (this subsystem → others):

| Module imported | Subsystem |
|---|---|
| `log_style` (all files), `diagnostics.voice_budget` (discord_player), `bus.envelope.TurnScope` (tts_player, type-only) | 01 |
| `config` (DeepgramSTTConfig/STTConfig/…/TTSConfig/LocalTurnConfig, type + runtime) | 02 |
| `llm.Message` (stt/protocol `to_message`) | 08 |

Imported by (others → this subsystem):

| Importer | Subsystem | What |
|---|---|---|
| `bot.py` | 10 | DaveVoiceClient, RecordingSink, DEFAULT_IDLE_FINALIZE_S, Transcriber/TranscriptionResult + UtteranceEndpointer (types) |
| `commands/run.py` | 10 | create_transcriber, create_tts_client, DiscordVoicePlayer/LoggingTTSPlayer/TTSPlayer, create_local_turn_detector |
| `familiar.py` | 02 | Transcriber / TTS-client union / LocalTurnDetector fields on `Familiar` |
| `sources/voice.py` | 09-adjacent (bus source; spec 01 groups it under 09, wiring in 10) | TranscriptionResult consumption, `record_vad_end` contract |
| `processors/voice_responder.py` | 06 | SentenceStreamer, TTSPlayer protocol |

External services: Discord voice gateway (WSS + UDP RTP, via py-cord →
songbird in Rust), Deepgram streaming STT, Cartesia TTS WS, Azure
Cognitive Services Speech, Google Gemini API, HuggingFace Hub (one-time
smart-turn ONNX download). Native/local runtimes: davey (libdave MLS),
ten-vad (native lib + bundled ONNX), onnxruntime, NeMo (torch),
faster-whisper (CTranslate2).

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `tests/test_dave_client.py` | TX encrypt layering + RTP header bytes, RX decrypt gating (no session / not ready / unknown SSRC / decrypt error drops frame) | needs-Rust-mock (fake DaveSession + ssrc_map) |
| `tests/test_dave_ws.py` | IDENTIFY includes dave version; opcode constants; binary dispatch + seq_ack; ops 25/27/29/30 flows incl. commit+welcome concat, error→op31+reinit; TX frame layout; ops 21/22/24 JSON flows; SESSION_DESCRIPTION version capture | needs-Rust-mock (fake WS + session) |
| `tests/test_recording_sink.py` | (user, mono) tuple via call_soon_threadsafe; Sink subclass; cleanup sets finished | needs-Rust-mock; Sink-subclass assertion is Python-specific-skip |
| `tests/test_voice_audio.py` | mono↔stereo math incl. byte-identical fallback, error cases; DISCORD_FRAME_SIZE=3840; StreamingPCMSource full read/EOS/partial-pad/block/unblock/cleanup + jitter (preroll first-read-only, EOS overrides, pad vs block) | logic-portable (thread-based blocking tests map to std::thread) |
| `tests/test_audio_resample.py` | 3:1 mean, carry across feeds, close zero-pads, reset, odd-length error, int16 edge numerics | logic-portable |
| `tests/test_ten_vad.py` | ctor validation (rate/hop), exact-hop chunk check, wrapper-side thresholding, reset rebuilds handle | needs-Rust-mock (native handle stubbed) |
| `tests/test_smart_turn.py` | softmax vs sigmoid output shapes, float32 /32768 normalization, 16 s tail truncation, bad-shape ValueError | needs-Rust-mock (ORT session stubbed) |
| `tests/test_utterance_endpointer.py` | every state edge: idle-silence no-classify, speech→silence classify+callback, incomplete holds, extended silence no-refire, force-complete (4 cases incl. fires-on-state-with-drained-buffer, no-refire-after-normal-completion), reset, SmartTurn off-thread, sub-frame carry | logic-portable (off-thread assertion becomes spawn_blocking check) |
| `tests/test_endpointer_audio_fixtures.py` | real resampler+framer with synthesized 48 k PCM: complete-sentence (1 classify, callback==classified buffer, ≥60 % of speech bytes), mid-thought pauses < silence_ms don't classify, filler incomplete→complete (2 classifies, 1 callback, 2nd buffer larger), filler+silence no-refire | logic-portable |
| `tests/test_transcription.py` | TranscriptionResult fields + to_message naming; DG URL param set/order-independent presence, interims gating of utterance_end_ms, keyterms repeat, clone fidelity; parse-response cases; lifecycle (send-before-start raises, closed-ws skips, finalize JSON, stop idempotent, results→queue); reconnect classification (shutdown/auth-4xxx/1008 stop; max attempts), replay buffer (buffer-when-closed, replay-then-delayed-Finalize, trim), backoff sequence 0/1/2/4…cap, keepalive (interval, survives errors, follows reconnect, cancelled on stop) | needs-Rust-mock (fake WS); logic (parse/url/backoff) portable |
| `tests/test_stt_factory.py` | protocol conformance of all 3 backends; dispatch; missing-key ValueError; unknown backend | logic-portable; runtime_checkable-Protocol checks become trait-impl compile checks |
| `tests/test_parakeet_transcriber.py` / `test_faster_whisper_transcriber.py` | buffer-and-finalize semantics: resample+buffer, finalize emits one final + clears, empty-buffer noop, empty-text skip, clone shares model + idle_close, lazy load once, stop clears, both NeMo shapes / segment-join + language passthrough / f32 dtype, NO endpointing_ms attr | needs-Rust-mock (model handle stubbed); likely replaced by whisper-rs equivalent |
| `tests/test_turn_detection_factory.py` | detector built from downloaded path; None on download failure / missing path; knob passthrough | needs-Rust-mock (hf download stub) |
| `tests/test_tts.py` | Cartesia payload/url/headers, chunk concat, ts secs→ms, error/close raise; stream chunk order, empty-skip, ts-dropped; greeting cache hit/miss/key-isolation; Azure ticks→ms, non-word skip, failure raise, executor; Azure stream: chunk order, COPIED buffers, start_speaking used, cancel raise, early-close unblock, mid-stream cancel raise vs clean EOF; upsample math; ts estimation; style prompt composition; all factory env/field errors incl. GOOGLE>GEMINI key priority | mostly needs-Rust-mock (fake WS/SDK); math + composition logic-portable |
| `tests/test_discord_voice_player.py` | buffered synth+play, stereo conversion, skip paths, barge-in vc.stop, play-lock serialization + cancel-then-speak no-collision, budget stamp; streaming: path selection via hasattr, jitter attr passthrough, stereo feed, play-before-drain, cancel mid-stream, empty-stream/first-chunk-error no-play | needs-Rust-mock (vc + stub TTS); hasattr-dispatch becomes trait method |
| `tests/test_tts_player_mock.py` | mock pacing, prompt cancel, stop flush | logic-portable (test util) |
| `tests/test_sentence_streamer.py` | boundary rules, abbreviations, initials, flush semantics, `<silent>` never emits, `?!` collapse, newline as space | logic-portable |
| `tests/test_voice_intake.py` | 10-layer pump contracts THIS subsystem must satisfy: clone-per-user, endpointing_ms=10 poke, fork to endpointer, endpointer-complete→finalize + vad_end, idle finalize (0.5 s vs endpointer fallback), watchdog close/reopen, user_id stamping, parallel clone stop | needs-Rust-mock; belongs to 10 but is the integration oracle for 09 seams |

## Rust port notes

- **Crate seams.** Natural split: `audio-core` (resampler, mono/stereo,
  StreamingPCMSource-equivalent ring buffer, SentenceStreamer — all pure,
  fully test-portable), `turn-detect` (VAD trait + endpointer state
  machine + ORT-backed SmartTurn), `stt` (Transcriber trait + Deepgram +
  optional local backends), `tts` (client trait + 3 impls + player).
  Suggested crates: `tokio`, `tokio-tungstenite` (Deepgram/Cartesia WS),
  `serde`/`serde_json`, `base64`, `sha2`, `ort` (onnxruntime bindings),
  `hf-hub` (weights download), `songbird` (Discord voice), `reqwest`.
- **DAVE is the highest-risk area.** py-cord + `davey` (libdave bindings)
  is a bespoke stack; `songbird` has its own voice gateway/UDP loop and
  (verify at port time) DAVE support status. Porting `DaveVoiceWebSocket`
  1:1 only makes sense if the Rust Discord stack exposes the same
  subclass seams; otherwise reimplement invariants A.1–A.14 inside
  songbird's driver hooks and bind libdave (C++) via FFI or use an MLS
  crate (`openmls`) if a Rust DAVE implementation exists. Budget for a
  spike.
- **Threading model maps cleanly.** py-cord's recording thread →
  songbird's event-driven receive (already gives per-SSRC PCM — the
  RecordingSink/`call_soon_threadsafe` bridge may vanish entirely).
  Pycord's 20 ms audio thread → songbird `Input`/`AudioStream`;
  `StreamingPCMSource` becomes an `impl Read`/`Input` over an
  `Arc<(Mutex<VecDeque<u8>>, Condvar)>` — keep invariants D.20–22 (EOS
  overrides, final-frame zero-pad, `b""` termination) exactly.
- **Protocol → trait mapping.** `Transcriber` and `TTSPlayer` become
  traits (`async fn` in trait / `#[async_trait]`). The duck-typed TTS
  client surface (`hasattr(synthesize_stream)`, jitter attrs) must be
  redesigned: a `TtsClient` trait with
  `fn streaming(&self) -> Option<&dyn StreamingTts>` (or an enum) plus
  `fn jitter_hints(&self) -> JitterHints` — do NOT emulate getattr.
  Same for the Deepgram-only `endpointing_ms` poke (J.70): make it a
  builder/setter on the Deepgram type and let the wiring downcast or use
  a dedicated trait method with a default no-op.
- **asyncio → tokio.** `asyncio.Lock` play lock → `tokio::sync::Mutex`
  (fairness: tokio Mutex is FIFO — matches). `asyncio.to_thread` (ONNX,
  NeMo, model load) → `spawn_blocking`. `call_soon_threadsafe` bridges →
  `mpsc` channels. Deepgram's receive/keepalive task pair → two spawned
  tasks with a shared `watch`/AtomicBool for `_closing`/`_shutting_down`;
  Python's benign task-cancellation-with-suppress patterns become
  `JoinHandle::abort()` + await. Async generators
  (`synthesize_stream`) → `impl Stream<Item = Result<Bytes>>`; the
  early-`aclose` cleanup (Cartesia WS close, Azure stop+join) must move
  into `Drop`/explicit `close()` since Rust streams have no finally-on-
  drop-of-consumer guarantee — recommend an owned struct with `Drop`.
- **Cancellation is cooperative, never preemptive.** `TurnScope.
  is_cancelled()` polling (20 ms in the player, per-chunk in drains, per
  tick in mock players) should become a `CancellationToken`; preserve the
  bounded stop-drain (H.63) — it exists to serialize against the audio
  thread's tick, which songbird may render unnecessary (re-verify the
  'Already playing' race under songbird and keep the test).
- **Integer math parity.** Floor-division semantics (B.17) and the
  `int()` truncation in endpointer thresholds (E.25) are pinned by
  byte-exact tests; use `div_euclid` and explicit truncation.
- **Local STT backends are optional and replaceable.** NeMo/Parakeet has
  no Rust runtime — plan to substitute `whisper-rs` or sherpa-onnx behind
  the same buffer-and-finalize contract (F.43–46) rather than porting
  Parakeet; keep the contract tests, swap the engine. `ten-vad` ships a C
  API — bindgen it, or substitute Silero-ONNX via `ort` behind the VAD
  trait (threshold + 256-sample hop contract preserved).
- **Do not port**: numpy-less fallback paths (single impl suffices), the
  lazy-import machinery (Cargo features replace extras: `local-turn`,
  `local-stt`), `MockTTSPlayer`/`LoggingTTSPlayer` pacing quirks beyond
  what tests need, and the `# ty: ignore` shims. DO port the log lines'
  structured keys (`decision=`, `close_code=`, replay counts) — ops
  workflows grep them.
