# DAVE integration-testing runbook (Rust port)

Validate the live-Discord voice path and the local-ML features on your own
machine. Work top-to-bottom; each smoke stage has do / expect / if-it-fails.
Per-user familiars resolve from the platform data dir (`FAMILIARS_ROOT`
overrides); the tracked `_default` skeleton resolves from
`data/familiars/_default` (`FAMILIAR_DEFAULTS_ROOT` overrides). Repo-relative
paths in this doc therefore refer to `_default` and build artifacts only —
see [On-disk layout](../getting-started/on-disk-layout.md#where-the-familiars-root-lives).

> **READ THIS FIRST — the voice glue is now wired; validate it live.**
> The composition-root wiring gaps are closed (parity-audit §3c + runbook flags).
> Concretely, in this tree:
> - `/subscribe-voice` and `/unsubscribe-voice` are **registered and dispatched**
>   under `discord-voice` (`src/bot.rs`: `Handler::ready` command list +
>   `dispatch_subscribe_voice` / `dispatch_unsubscribe_voice`).
> - `join_voice` → `start_voice_intake` runs on subscribe (songbird join +
>   DAVE, `RecordingSink` attach, per-speaker pumps + `VoiceSource`);
>   `stop_voice_intake` runs on unsubscribe + shutdown. `BotHandle::voice_runtime`
>   is populated per voice channel.
> - `DiscordVoicePlayer`'s voice-client getter now reads `handle.voice_runtime`
>   (`src/commands/run.rs`), returning the channel's live songbird-backed
>   `VoiceClientLike`, so TTS playback reaches the call.
> - **Still open:** TEN-VAD has **no native backend**: `TenVad::new` always
>   returns `MissingBackend` (`src/voice/turn_detection/ten_vad.rs`), so the
>   `ten+smart_turn` endpointer cannot be built at runtime and silently degrades
>   to Deepgram idle-finalize.
>
> The songbird playback bridge (`SongbirdVoiceClient`) and slash dispatch are
> `discord-voice` build + clippy clean but have **not** been exercised against a
> live Discord voice channel — that is exactly what Stages b-f below now do.
> **What you CAN validate without a call:** Stage 0-1 (text path end-to-end) and
> Section 5 (local-ML tests). Stages b-f now run as written; TEN-VAD live
> endpointing remains degraded until its FFI backend lands.

---

## 1. Prereqs

### Build

Toolchain: stable rustc (workspace pins `edition = 2024`, `rust-version = 1.85`).
Cargo `--features` are additive over the default set (`store, net, images`).

```bash
# Text-only bot (Stage 0-1): default features already include net for Deepgram HTTP.
cargo build --release --features discord

# Voice + Deepgram STT (Stages b-f, once glue lands). `discord-voice` implies
# `discord`; `stt-deepgram` implies `net`. --release matters: real-time 48 kHz
# audio under a debug build is often too slow.
cargo build --release --features discord-voice,stt-deepgram
```

`discord`, `discord-voice`, and `discord-voice,stt-deepgram` all type-check clean
on stable (verified). `local-embed` / `local-turn` additionally need a native
onnxruntime (Section 5).

**Windows note (voice builds):** `discord-voice` pulls `songbird → opus2 →
libopus_sys`, which compiles libopus **from source with CMake**. If the build
fails with `is 'cmake' not installed?`:

```powershell
winget install Kitware.CMake   # then open a NEW shell so PATH updates
```

CMake drives the MSVC toolset you already have (the error appearing at the
CMake step, not a compiler step, means VS Build Tools are fine). Alternative if
you'd rather not install CMake: point `OPUS_LIB_DIR` at a prebuilt static
libopus and rebuild. Text-only builds (`--features discord`) never need CMake.

### Env vars (`.env` at the repo root is autoloaded by `dotenvy`)

Copy `.env.example` to `.env` and fill:

| Var | Required for | Notes |
|---|---|---|
| `DISCORD_BOT` | always | Bot token. **Not** `DISCORD_BOT_TOKEN`. Missing → exit 1. |
| `OPENROUTER_API_KEY` | always | LLM. Missing → exit 1. |
| `DEEPGRAM_API_KEY` | voice STT | Required when `[providers.stt].backend="deepgram"` (default). |
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` | TTS (default provider) | Or switch provider below. |
| `CARTESIA_API_KEY` | TTS if `[tts].provider="cartesia"` | Enables the byte-streaming playback path. |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | TTS if `[tts].provider="gemini"` | `GOOGLE_` wins if both set. |
| `FAMILIAR_ID` | selects familiar | Or pass `--familiar <id>` (flag wins). |
| `FAMILIARS_ROOT` | per-user familiars root | Overrides the platform data-dir default (#201). |
| `FAMILIAR_DEFAULTS_ROOT` | `_default` skeleton root | Overrides the CWD-relative `data/familiars`. |

TTS/STT/turn-detector construction **degrades, never fails**: an unavailable key
logs a warning and the text path keeps working (`run.rs` L410-429).

### Discord Developer Portal

- Enable **Message Content Intent** (privileged). The bot requests
  `MESSAGE_CONTENT | GUILD_VOICE_STATES | GUILD_MESSAGE_TYPING` plus
  non-privileged (`bot.rs` L1917-1920). Without Message Content the bot logs in
  but sees no message text.
- **`GUILD_MEMBERS` is deliberately NOT requested.** Member resolution is
  cache-only (see Known Sharp Edges).
- Invite scopes: `bot` + `applications.commands`; permissions: Send Messages,
  Read Message History, Add Reactions, Connect, Speak.

### Familiar directory shape

Per-user familiars resolve from the **platform data dir**
(`~/.local/share/familiar-connect/familiars` on Linux; macOS/Windows
equivalents), independent of CWD. `FAMILIARS_ROOT` overrides the root (top
precedence). The tracked `_default` skeleton is a repo resource resolved
separately from `data/familiars/_default` (`FAMILIAR_DEFAULTS_ROOT` overrides).
On startup a one-shot, idempotent, never-clobber migration moves any legacy
`./data/familiars/<id>` (other than `_default`) into the resolved root. See
[On-disk layout](../getting-started/on-disk-layout.md#where-the-familiars-root-lives).

To keep everything inside the repo checkout while smoke-testing, point the root
at `data/familiars`:

```powershell
$env:FAMILIARS_ROOT = "data/familiars"
cargo run --release --features discord -- run --familiar <id> -v
```

A missing familiar errors with the **absolute** path it checked, so a
root/CWD mismatch is self-diagnosing.

The `<root>/<id>/` folder must exist (`run.rs` `resolve_familiar_root` only
checks existence). Config = `<root>/<id>/character.toml` **deep-merged over**
`data/familiars/_default/character.toml` (a missing per-familiar `character.toml`
just inherits the defaults). Minimum useful setup (in-repo root):

```bash
export FAMILIARS_ROOT=data/familiars
cp -r data/familiars/_default data/familiars/aria
# edit data/familiars/aria/character.md  (the persona; CharacterCardLayer reads it)
# character.toml already ships [llm.fast] [llm.prose] [llm.background] +
# [providers.stt] [tts] [providers.turn_detection] — the required trio of LLM
# slots must be present or run exits 1.
```

With `FAMILIARS_ROOT=data/familiars`, `_default` itself is runnable as a smoke
target (`--familiar _default`) since it carries `character.toml` + `character.md`.

### Run

```bash
cargo run --release --features discord-voice,stt-deepgram -- run --familiar aria -v
```

CLI is `familiar-connect <run|diagnose|version>` (no `sleep` verb). `-v` = INFO
(needed to see the voice/decision log lines below), `-vv` = DEBUG for **all**
targets incl. `songbird` and `serenity`.

---

## 2. Smoke ladder

### Stage 0 — text-only start (config + LLM + gateway)

**Do:** `cargo run --release --features discord -- run --familiar aria -v`, then
`/subscribe-text` in a guild channel and send a message that pings the bot.
**Expect:** login; `/subscribe-text` replies "Listening in this channel."; the
bot streams a reply. Presence shows `✨ <guild> -> <channel>`.
**If it fails:** `DISCORD_BOT` unset → "environment variable is not set";
familiar missing → "Familiar folder does not exist"; `OPENROUTER_API_KEY` unset →
exit 1; no reply → check Message Content Intent is enabled.

### Stage 1 — Deepgram + TTS construction (no voice yet)

**Do:** start with `--features discord-voice,stt-deepgram -v` and
`DEEPGRAM_API_KEY` + a TTS key set.
**Expect:** startup logs no `Transcriber unavailable` / `TTS client unavailable`
warnings. (The transcriber template is built; it isn't connected until a speaker
is demuxed.)
**If it fails:** a warning naming the missing key — the bot still runs text-only.
This proves the STT/TTS factories resolve before you attempt voice.

### Stage 2 (b) — voice join + DAVE handshake, single speaker

> The `/subscribe-voice` → `join_voice` → `start_voice_intake` glue is landed
> (build + clippy clean, live-untested). Validate:

**Do:** join a voice channel, run `/subscribe-voice`.
**Expect:** reply `Joined <channel>.`; log `[🎙️  Voice] intake=started
channel=<id>`. DAVE/MLS is **owned entirely by songbird 0.6** (davey/OpenMLS) —
there is no app-level handshake code; `voice/dave_client.rs` and `dave_ws.rs` are
doc-only checklists. Confirm the MLS handshake with `-vv` and
`grep -i 'dave\|mls\|songbird::driver'`.
**If it fails:** gateway close `4017` = songbird didn't advertise
`max_dave_protocol_version` (A.1); `4005` = binary-frame seq framing (A.5);
close-loop = seq_ack desync (A.4). All are songbird-internal — capture `-vv`
songbird logs and file upstream.

### Stage 3 (c) — STT round-trip (speak → transcript → reply → TTS)

> Needs Stage b; the `DiscordVoicePlayer` voice-client seam now reads
> `voice_runtime` (was `|| None`) so audio plays back through the songbird call.

**Do:** speak one sentence, pause ~1 s.
**Expect (in order):** `[🎙️  Voice] user=<id> transcriber=opened`; a
`voice.transcript.final` (visible via debug-logger at `-vv`); `[Voice]
decision=respond turn=voice-...`; audible reply.
**If it fails:** transcript only after your *next* utterance → idle-finalize not
firing (see Sharp Edges); `decision=silent` (`[💤 Voice]`) → `<silent>` sentinel
latched, expected for filler; no audio out → the `|| None` voice-client seam.

### Stage 4 (d) — MULTI-SPEAKER per-SSRC receive  ← #1 residual risk

The ~3-month-old songbird DAVE receive path. Two people in the channel.
**Do:** both speak, overlapping and taking turns.
**Expect:** two `transcriber=opened` lines (one per user id); each utterance
attributed to the right speaker (distinct `user_id` on each
`voice.transcript.final`); independent reply pipelines (per-`(channel,user)` turn
scopes — the two never cancel each other).
**If it fails:** one speaker's audio decrypts, the other silent → per-SSRC MLS
decrypt gap; wrong attribution → SSRC→user map stale (`SpeakingStateUpdate`
inconsistency — Sharp Edges). Capture `-vv` and both SSRCs.

### Stage 5 (e) — barge-in during playback (<200 ms)

> **BLOCKER:** needs playback (Stage c).

**Do:** while the bot is speaking, start talking.
**Expect:** in-flight audio cut within ~one 20 ms poll tick; `[Voice]
decision=preempted turn=...` for the cancelled turn; a fresh reply to your new
utterance. Same-speaker self-barge cancels via the scope; cross-speaker does
**not** cut the other's reply (shared voice client).
**If it fails:** `ClientException('Already playing audio.')`-class collision on
the immediate next speaker → the stop-drain window regressed.

### Stage 6 (f) — DAVE epoch transitions (member join/leave re-keys MLS)

**Do:** with a call live and audio flowing, have a third person join, then leave.
**Expect:** audio survives both. songbird processes PROPOSALS/COMMIT/WELCOME
(ops 27-30) and re-keys; a brief first-syllable drop in the ratchet window right
after the transition is normal (Sharp Edges). Transcripts resume within ~1 s.
**If it fails:** call drops on join/leave → commit/welcome recovery (A.11) failed;
capture `-vv`, `grep -iE 'welcome|commit|epoch|invalid'`.

---

## 3. The A.1–A.14 verification checklist

DAVE lives inside songbird; the app carries **no** DAVE runtime code, so these are
verified by outcome + songbird `-vv` logs, not by app-level greps. The invariant
text is the doc checklist in `voice/dave_ws.rs` (A.1-A.12) and `dave_client.rs`
(A.13-A.14).

| # | Invariant | Observable pass criterion |
|---|---|---|
| A.1 | IDENTIFY carries `max_dave_protocol_version` | Join succeeds; **no** gateway close `4017`. |
| A.2 | Negotiate → send MLS key package (op 26) after `secret_key`, only when version>0 | Call reaches connected; `-vv` shows key-package TX. |
| A.3 | poll_event dispatches text **and** binary frames | Binary MLS frames processed; connection stable (py-cord dropped them — songbird must not). |
| A.4 | seq_ack tracks binary frames before dispatch | No resume/close-loop after binary frames. |
| A.5 | Binary TX frames omit the seq prefix (`[op][payload]`) | No close `4005`. |
| A.6 | SESSION_DESCRIPTION reads version before session init | First binary MLS frame processed without desync. |
| A.7 | op 21 PREPARE_TRANSITION → store id, passthrough on downgrade-to-0, reply op 23 | Downgrade/transition doesn't drop the call. |
| A.8 | op 22 EXECUTE_TRANSITION → apply/ignore unknown id | Initial handshake completes (unknown id is normal there). |
| A.9 | op 24 PREPARE_EPOCH → reinit only at epoch==1 w/ no session | No double-key-package desync on connect. |
| A.10 | op 25/27 external-sender + proposals → reply op 28 commit(+welcome) | Member add/remove re-keys cleanly (Stage f). |
| A.11 | ops 29/30 commit/welcome; failure → op 31 + session reinit (recover) | Bad commit recovers instead of crashing (Stage f). |
| A.12 | Every binary handler length-checks; missing session = warn-drop | No panic/crash on malformed frame; call survives. |
| A.13 | RX: gate on `data[1]&0x78==0x78`, drop pause/silence-sentinel, decrypt `(user_id, audio)`, unknown SSRC passes through | Correct per-speaker transcripts (Stage c/d); no decrypt-error crash. |
| A.14 | TX: opus → DAVE `encrypt_opus` (ready only) → RTP header → SRTP | Peers hear the bot (needs playback seam, Stage c). |

If any observable fails, the fix is in **songbird**, not this repo. Capture `-vv`
and open an upstream issue; there were no open DAVE issues at 0.6.0 release.

---

## 4. Known sharp edges (expect these; don't panic)

- **Ratchet-window packet drops.** The first instants of audio right after a join
  or an epoch transition are silently dropped until key ratchets establish. Lose a
  first syllable — inherent to DAVE key-ratcheting. Not a bug.
- **Benign songbird receive-path log noise (#199).** At `-vv`, songbird emits
  lines like `RTCP decryption failed`, `opus_decode InvalidPacket`, and
  `Decode error for SSRC <n>` during normal operation — expected under DAVE's
  per-SSRC decrypt/ratchet behavior, not data loss. Do **not** file these
  against this repo; the fix (if any) is upstream in songbird. #199 was the
  phantom-bug chase that concluded exactly this.
- **`SpeakingStateUpdate` fired inconsistently** (songbird PR #291 caveat). The
  SSRC→user map (`SsrcMap`, fed by those events) can lag. The design does **not**
  key per-user state solely off speaking events — first-audio-chunk lazy creation
  + the idle-finalize fallback cover it. A momentarily-`None` user id passes audio
  through undecrypted rather than crashing.
- **Cache-only member resolution (`resolve_member` = `|| voice_member_cached`).**
  No `GUILD_MEMBERS` intent + no REST fetch on the audio path, so a voice-only
  joiner who hasn't typed or triggered a voice-state update resolves to `None` →
  **anonymous voice turns** until they type or their state updates. Expected; not
  data loss.
- **Local turn detection silently degrades.** With `strategy="ten+smart_turn"`,
  `create_local_turn_detector` downloads Smart Turn weights and builds the
  detector, but the per-user `make_endpointer` call fails (`TenVad::new` →
  `MissingBackend`), so endpointing falls back to Deepgram idle-finalize (0.5 s).
  Look for `local_turn_detection=enabled` at startup **and** absence of live
  endpointer behavior. TEN-VAD FFI (`ten-vad-sys`) is not linked yet.
- **Idle-finalize is the trailing-silence backstop.** Discord halts RTP during
  silence, so neither Deepgram's endpointer nor the local chain sees turn-ending
  silence. The per-user pump force-finalizes after 0.5 s (plain Deepgram) /
  `idle_fallback_s` 1.5 s (local). "Transcript only arrives on my next sentence"
  means this backstop isn't arming.

---

## 5. Local-ML validation

Native onnxruntime is required (both features pull `ort`/fastembed). If your box
lacks it, install onnxruntime or let `ort` fetch a binary; a link error here is
environment, not code.

```bash
# Fast: scripted-model unit tests (no downloads). Smart Turn + TEN-VAD wrappers.
cargo test --features local-turn                 # smart_turn::tests, ten_vad::tests
cargo test --features local-embed                # fastembed::tests (stub loader)
```

**Real-weights tests are `#[ignore]`d — run explicitly with `-- --ignored`.**
There is exactly **one** in the tree:

| Test | Location | Command | Downloads | "Pass" |
|---|---|---|---|---|
| `real_bge_small_embeds` | `src/embedding/fastembed.rs:484` | `cargo test --features local-embed -- --ignored real_bge_small_embeds` | BGE-small-en-v1.5 ONNX (~130 MB) into fastembed's model cache | Embeds a batch, returns non-empty vectors of the model's dim; a few seconds after the one-time download. |

**Smart Turn has no `#[ignore]`d real-model test** — its unit tests use injected
`SmartTurnModel` doubles and run under plain `cargo test --features local-turn`.
The real ONNX path (`OrtSmartTurnModel::load`) is only exercised at runtime.

**Smart Turn weights (runtime, when strategy=`ten+smart_turn`):** repo
`pipecat-ai/smart-turn-v3`, file `smart-turn-v3.2-cpu.onnx` (~360 MB), fetched via
`hf-hub`'s sync API into the **HuggingFace Hub cache** (`~/.cache/huggingface/hub`,
override with `HF_HOME`) — **not** `data/models/`. Offline reruns hit the cache.
Startup logs `local_turn_detection=enabled vad=ten-vad smart_turn=<file> …`
(target `familiar_connect.voice.turn_detection.factory`) on success, or
`disabled reason=smart_turn_download_failed|smart_turn_cache_missing|huggingface_hub_missing`.

---

## 6. Escape hatches

**Capture more on failure.** Verbosity is the *only* log control — the subscriber
builds its filter from `-v` count and **ignores `RUST_LOG`** (`cli.rs` L207-212).
- `-v` → INFO (voice intake, decision lines, LLM call lines).
- `-vv` → DEBUG for every target, incl. `songbird` and `serenity` (the DAVE/MLS
  handshake, RTP, gateway opcodes). Use this for any Stage b-f failure.
- Redirect to a file: `... run --familiar aria -vv 2> voice.log`.

**Grep-able signals** (INFO unless noted): `intake=started` / `transcriber=opened`
/ `intake=stopped` (voice pump); `decision=respond|silent|preempted turn=…` (voice
responder); `local_turn_detection=enabled|disabled` (turn factory); `close_code=`
(Deepgram reconnect classification, `stt::deepgram`); `[Shutdown] … draining|clean`.

**Diagnose over captured logs.** Spans are logged as `span=<name> ms=<int>
status=<ok|error>`; aggregate them:
```bash
cargo run --release -- diagnose voice.log      # p50/p95/last_ms per span; '-' reads stdin
```
`/diagnostics` in Discord shows the same table live (plus focus + unread lines).
Voice budget spans to watch: `voice.total` (stt_final→playback_start),
`voice.vad_to_stt`, `voice.stt_to_ttft`, `voice.tts_to_playback`.

**Fallback seams if a dependency misbehaves:**
- **SQLite:** `rusqlite` (bundled) is already the default `store` engine — no
  turso beta in play; nothing to switch.
- **songbird:** pinned to `0.6.0` (DAVE mandatory, receive path ~3 months old).
  Keep it pinned; track point releases for receive-path fixes rather than
  floating the version.
- **STT:** an unavailable `DEEPGRAM_API_KEY` degrades to text-only (warning, not
  crash) — you never lose the gateway.
- **Local endpointing:** if `ten+smart_turn` misbehaves, set
  `[providers.turn_detection].strategy="deepgram"` to fall back to the hosted
  endpointer with zero code change.
