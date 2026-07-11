# Rust crate ecosystem report — Familiar-Connect Python→Rust rewrite

Research snapshot: 2026-07-11. Versions verified on crates.io / GitHub unless noted.
This document is the dependency contract for the port: porting agents pick crates
from here, not from training memory.

## Verdict on DAVE E2EE (the former blocker): portable in pure Rust, no FFI

**Songbird v0.6.0 "Hoopoe" (2026-04-05) ships DAVE E2EE support, built on the
pure-Rust `davey` crate — the same implementation the Python bot already uses
underneath** (the `davey` PyPI package is Snazzah's Rust crate with Python
bindings). The rewrite *removes* a binding layer rather than adding one.
`dave_client.py` / `dave_ws.py` are deleted, not ported: songbird owns the MLS
handshake and voice-gateway opcodes 21–31 internally.

Evidence:
- [songbird PR #291 "DAVE support"](https://github.com/serenity-rs/songbird/pull/291),
  merged 2026-03-28, closing [issue #293](https://github.com/serenity-rs/songbird/issues/293)
  (the 4017-style rejection referenced in `dave_ws.py`). Released in
  [v0.6.0](https://github.com/serenity-rs/songbird/releases): DAVE is mandatory
  for all 2026 voice connections and always-on in the `driver` feature
  (`davey ^0.1.2` hard dependency).
- `davey` crate: 0.1.4 (2026-06-22), actively maintained
  ([github.com/Snazzah/davey](https://github.com/Snazzah/davey), MLS via OpenMLS, MIT).
- **Receive under DAVE works but is the least-hardened path** (PR #291: sending
  "mostly works"; "Receiving is not very heavily tested"). Known sharp edges:
  packets arriving before key ratchets are established are silently dropped
  (first instants of audio after join/transition), and `SpeakingStateUpdate`
  events "are apparently not fired consistently". No open DAVE issues in
  songbird since the 0.6.0 release (checked 2026-07-11).

**Mitigations to carry into the port:** (a) accept first-syllable loss in the
ratchet window (same loss class exists today); (b) do not key per-user state
solely off speaking events — keep the idle-finalize fallback; (c) pin songbird
and track point releases; (d) **spike early**: join a DAVE call and verify
multi-speaker per-SSRC receive before building on it.

**Voice receive (per-SSRC):** mature, pre-dates DAVE. Opt-in `receive` cargo
feature: per-user jitter buffers, `VoiceTick` events deliver decoded PCM per
SSRC with speaking-state → user-id mapping. Matches the `RecordingSink`
per-user demux design directly. Songbird decodes/encodes Opus internally
(`opus2` 0.4.0; `audiopus` is unmaintained — avoid) and already depends on
`rubato` for resampling.

**Discord library: serenity 0.12.5 + songbird 0.6.0 (default features +
`receive`).** Serenity's batteries-included model (event handler trait, cache,
command framework) maps more directly from py-cord than twilight's
assemble-it-yourself crates. Slash commands, typing indicators
(`broadcast_typing`), presence updates, voice-state events all covered.

Risk: **LOW-MEDIUM** (receive-path maturity under DAVE is ~3 months old in the
wild).

> **CORRECTIONS (2026-07-11, scaffold build — see DESIGN.md D18):** two versions
> below failed empirical validation. `rusqlite` is pinned to **0.37** (0.40.1
> pulls libsqlite3-sys 0.38.1 whose build script needs unstable `cfg_select!`,
> won't compile on stable rustc 1.94.1). `rubato 4.0.0` **does not exist** —
> latest is **0.16.x** (also unifies with songbird's dependency tree). The
> Cargo.toml manifest in DESIGN.md §6 is authoritative over this report.

## Per-capability picks

| # | Capability | Pick | Version | Risk | Notes |
|---|---|---|---|---|---|
| 1 | Discord + DAVE voice | serenity + songbird (`davey` under hood) | 0.12.5 / 0.6.0 / 0.1.4 | Low-Med | See verdict above |
| 2 | Deepgram streaming STT | `deepgram` | 0.10.0 | Low | Official-org SDK. Live WS streaming verified in source: `endpointing(Endpointing)`, `utterance_end_ms(u16)`, `interim_results`, `no_delay`, `vad_events`, `keep_alive`, plus `finalize()` / `close_stream()` on the handle — clone-from-template + Finalize-driven flush ports 1:1. `keyterms()` (nova-3) and `keywords()` both present. |
| 3 | Turso/SQLite | `turso` (fallback: `rusqlite`) | 0.6.1 / 0.40.1 | Med / Low | `turso` is the same engine `pyturso` binds — async-native, SQLite file-format compatible; existing `history.db` opens unchanged. Still explicitly BETA upstream. `rusqlite` 0.40.1 reads the same files (sync — wrap in `spawn_blocking`); it is the escape hatch. `libsql` is a different lineage — skip. |
| 4 | Full-text search | `tantivy` | 0.26.1 | Low | Python side uses tantivy-py at the same 0.26 core — index format and API map directly; existing indexes open as-is. Keep versions matched. |
| 5 | Embeddings / ONNX | `fastembed` / `ort` | 5.17.2 / 2.0.0-rc.12 | Low / Med | fastembed-rs default model IS BGE-small-en-v1.5 (same ONNX weights). `ort` 2.0 is production-ready but not API-stable (rc.12) — expect churn at 2.0 final. Smart Turn v3 = plain ONNX via `ort`; port the existing dual-output-shape handling and verify input preprocessing. |
| 6 | OpenRouter LLM (SSE + tools) | `reqwest` + `eventsource-stream` | 0.2.3 | Low | Port the hand-rolled client (Python is httpx+SSE already). OpenRouter extras (`usage: {include:true}`, `provider`, cached-token counts) aren't modeled by typed OpenAI clients. Alt: `async-openai` 0.41.1 with base-URL override + "byot" mode. Skip `openrouter-rs` (too small). |
| 7a | Azure TTS | `azure-speech` 0.10.0 or REST via `reqwest` | — | Med | Pure-Rust WS reimplementation of the Speech SDK protocol; slower maintenance (last release ~Aug 2025) — budget for vendoring. The Python client only uses the buffered path, so plain REST also suffices. Decide at port time. |
| 7b | Cartesia TTS | custom `tokio-tungstenite` WS client | — | Low-Med | No official Rust SDK (crates.io `cartesia` is a dead placeholder). Python `synthesize_stream` is already a hand-rolled WS client — port it. Protocol: docs.cartesia.ai/api-reference/tts/websocket. |
| 7c | Gemini TTS | REST via `reqwest` | — | Low | No SDK needed. |
| 8 | Twitch EventSub | `twitch_api` | 0.8.0 | Low-Med | `Event::parse_websocket`, transport types, Helix subscription helpers. We own the WS connect/reconnect/session-welcome loop (~100–200 lines of glue vs Python twitchAPI's managed client). |
| 9 | Audio plumbing | songbird internal + `rubato` | 4.0.0 | Low | Receive gives decoded PCM per user; send takes PCM via `RawAdapter` (48kHz stereo). 48k→16k mono decimation for VAD/STT: keep the 3:1 boxcar or use `rubato`. Standalone Opus if ever needed: `opus` 0.3.1. |
| 10 | TEN-VAD | FFI-bind libten_vad (recommended) | — | Med | The Python wheel ships a native C library, not just ONNX. Options: (a) thin `bindgen` wrapper over the small C API using upstream prebuilt libs — recommended; (b) ONNX via `ort` + reimplement the feature-extraction frontend (more work, needs reference validation); (c) swap to Silero (`voice_activity_detector` crate) + re-tune thresholds. |
| 11 | TOML + deep-merge | `toml` | 1.1.2 | Low | Merge `toml::Value` trees with ~30 lines of custom code — preserves exact Python override semantics. `toml` also writes (replaces tomli-w). |
| 12 | dotenv | `dotenvy` | 0.15.7 | Low | Done, stable. |
| 13 | Image resize/JPEG | `image` | 0.25.10 | Low | Resize + JPEG re-encode built in (replaces Pillow usage in view_image). |
| 14 | TaskGroup / cancel scopes | `tokio` + `tokio-util` | 0.7.18 | Low | `JoinSet`/`TaskTracker` ≈ asyncio TaskGroup; hierarchical `CancellationToken` ≈ `TurnScope`. **Cancellation is cooperative at await/`select!` points, not thrown like `CancelledError` — audit every cancel point during port.** |
| 15 | Logging | `tracing` + `tracing-subscriber` | 0.3.23 | Low | ANSI styling built in (replaces colorama + logging); tracing spans align with the `@span`/`SpanCollector` telemetry design. |

## Bounded custom work (all with working Python reference implementations in-repo)

1. TEN-VAD FFI binding (bindgen wrapper over libten_vad's C API).
2. Cartesia WebSocket TTS client port.
3. Azure TTS path (azure-speech crate vs plain REST) — buffered only.
4. Twitch EventSub WS glue loop.
5. OpenRouter SSE client (~few hundred lines, shape already exists in `llm.py`).
6. TOML deep-merge (~30 lines).

## Biggest open risks

1. **songbird DAVE-receive maturity** — spike first, before the port depends on it.
2. **turso beta label** — `rusqlite` is the escape hatch on the same file format.
3. **`ort` API instability** — pin to rc.12; isolate behind the existing embedding/VAD protocol seams.

Sources: songbird PR #291 / releases / issue #293 · Snazzah/davey · crates.io/davey ·
deepgram-rust-sdk · tursodatabase/turso · fastembed-rs · ort.pyke.io ·
OpenRouter structured-outputs docs · twitch_api eventsub docs ·
jbernavaprah/azure-speech-sdk-rs · docs.cartesia.ai TTS WebSocket ·
Discord blog "Bringing DAVE to All Platforms".
