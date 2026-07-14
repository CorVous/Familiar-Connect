//! DAVE voice-gateway WebSocket — **verification checklist**, not a 1:1 port
//! (subsystem 09; Python `voice/dave_ws.py`; DESIGN D8).
//!
//! In Python this subclasses py-cord's `DiscordVoiceWebSocket` to add DAVE
//! (Discord's MLS-based end-to-end encryption) opcode handling. In the Rust
//! stack **songbird owns the voice gateway, UDP loop, Opus, and DAVE/MLS**
//! (songbird ≥ 0.6 embeds `davey`/libdave), so there is no subclass seam to
//! transliterate. This module is therefore a documentation stub: the Python
//! invariants below are rewritten as a checklist the Layer-4 wiring verifies
//! against songbird's driver events/config hooks. The corresponding Python
//! tests (`test_dave_ws.py`) are recorded as skip-with-reason.
//!
//! ## Wire opcodes (the invariant "vocabulary")
//!
//! JSON ops: `21 PREPARE_TRANSITION {transition_id, protocol_version}`,
//! `22 EXECUTE_TRANSITION {transition_id}`, `23 TRANSITION_READY` (TX)
//! `{transition_id}`, `24 PREPARE_EPOCH {epoch, protocol_version}`,
//! `31 INVALID_COMMIT_WELCOME` (TX) `{transition_id}`.
//!
//! Binary RX frame: `[seq: u16 BE][op: u8][payload]`; binary TX frame:
//! `[op: u8][payload]` (**no** seq prefix). Binary ops: `25 EXTERNAL_SENDER`,
//! `26 KEY_PACKAGE` (TX), `27 PROPOSALS ([optype u8: 0=append else revoke]
//! [proposals])`, `28 COMMIT_WELCOME` (TX, `commit || welcome?`),
//! `29 ANNOUNCE_COMMIT_TRANSITION ([transition_id u16 BE][commit])`,
//! `30 WELCOME ([transition_id u16 BE][welcome])`.
//!
//! ## Verification checklist (spec 09 §A.1–A.12) against songbird
//!
//! - **A.1 IDENTIFY carries `max_dave_protocol_version`.** Omitting it closes the
//!   gateway with 4017. Verify songbird advertises the DAVE version in its voice
//!   IDENTIFY (driver config); if a knob is exposed, ensure it is not forced to 0.
//! - **A.3 poll_event dispatches BOTH text and binary frames.** py-cord silently
//!   drops binary; songbird's gateway must route binary MLS frames to its DAVE
//!   handler. ERROR frame → connection-closed; CLOSE/CLOSING/CLOSED → closed with
//!   the close code; 30 s receive timeout.
//! - **A.4 seq_ack tracks binary frames.** The 2-byte seq prefix is read and
//!   applied *before* dispatch; if binary frames don't advance `seq_ack`, gateway
//!   resume desyncs and close-loops. Verify songbird acks binary seq.
//! - **A.5 binary TX frames omit the seq prefix** (`[op][payload]`). Adding one
//!   shifts the opcode and triggers close 4005.
//! - **A.6 SESSION_DESCRIPTION reads `dave_protocol_version` before session init**
//!   so session setup precedes the first binary MLS frame; the base handler still
//!   runs so JSON `seq_ack` updates.
//! - **A.7 op 21 PREPARE_TRANSITION:** store `transition_id → protocol_version`;
//!   if downgrading to version 0 with a live session, enable passthrough
//!   (`set_passthrough_mode(true, 120)`); always reply op 23 TRANSITION_READY.
//! - **A.8 op 22 EXECUTE_TRANSITION:** pop pending; unknown id → debug-log +
//!   return (normal on the initial handshake); known → apply `protocol_version`.
//! - **A.9 op 24 PREPARE_EPOCH:** reinit the session only when `epoch == 1` AND no
//!   session exists — a second key package after SESSION_DESCRIPTION init
//!   desyncs and closes the connection.
//! - **A.10 op 25 EXTERNAL_SENDER / op 27 PROPOSALS:** install external sender;
//!   proposals first byte `0` → append, else revoke; `process_proposals` may
//!   yield nothing or a commit (+optional welcome) → reply op 28 with
//!   `commit || welcome` concatenated.
//! - **A.11 ops 29/30 (ANNOUNCE_COMMIT / WELCOME):** payload
//!   `[transition_id u16 BE][data]`; a `process_commit`/`process_welcome` failure
//!   → send op 31 INVALID_COMMIT_WELCOME with the transition id, then reinit the
//!   whole session (fresh key package) — recovery, not crash; success → op 23.
//! - **A.12 every binary handler length-checks its payload** and warn-drops short
//!   frames; a missing session is warn-drop, never a crash.
//!
//! Ops workflows grep the structured `decision=` / `close_code=` keys these
//! handlers log; preserve them wherever the songbird glue lands (spec 09 port
//! notes).

/// DAVE JSON gateway opcodes (documented for the Layer-4 songbird glue).
pub mod opcodes {
    /// PREPARE_TRANSITION `{transition_id, protocol_version}` (RX).
    pub const DAVE_PREPARE_TRANSITION: u8 = 21;
    /// EXECUTE_TRANSITION `{transition_id}` (RX).
    pub const DAVE_EXECUTE_TRANSITION: u8 = 22;
    /// TRANSITION_READY `{transition_id}` (TX).
    pub const DAVE_TRANSITION_READY: u8 = 23;
    /// PREPARE_EPOCH `{epoch, protocol_version}` (RX).
    pub const DAVE_PREPARE_EPOCH: u8 = 24;
    /// EXTERNAL_SENDER MLS credential (binary RX).
    pub const MLS_EXTERNAL_SENDER: u8 = 25;
    /// KEY_PACKAGE (binary TX).
    pub const MLS_KEY_PACKAGE: u8 = 26;
    /// PROPOSALS `[optype u8][proposals]` (binary RX).
    pub const MLS_PROPOSALS: u8 = 27;
    /// COMMIT_WELCOME `commit || welcome?` (binary TX).
    pub const MLS_COMMIT_WELCOME: u8 = 28;
    /// ANNOUNCE_COMMIT_TRANSITION `[transition_id u16 BE][commit]` (binary RX).
    pub const MLS_ANNOUNCE_COMMIT_TRANSITION: u8 = 29;
    /// WELCOME `[transition_id u16 BE][welcome]` (binary RX).
    pub const MLS_WELCOME: u8 = 30;
    /// INVALID_COMMIT_WELCOME `{transition_id}` (TX, recovery).
    pub const MLS_INVALID_COMMIT_WELCOME: u8 = 31;
}
