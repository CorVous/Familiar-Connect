//! DAVE E2EE voice client — **verification checklist**, not an implementation
//! (subsystem 09).
//!
//! **songbird owns the voice connection, Opus, SRTP, and DAVE/MLS**, so there is
//! no client type to implement here. This module is a documentation stub: the
//! RX/TX layering invariants below are the checklist the Layer-4 songbird glue
//! is verified against. Opcode/wire details live in [`super::dave_ws`].
//!
//! ## Session lifecycle
//!
//! - **A.2 connect + reinit.** Poll until the voice `secret_key` is set, then —
//!   only when the negotiated `dave_protocol_version > 0` (from
//!   SESSION_DESCRIPTION) — create a fresh session and send the serialized MLS
//!   key package as binary op 26. Recovery from an invalid commit/welcome reinits
//!   the whole session (fresh key package). Verify songbird performs this
//!   negotiate→key-package handshake and exposes a recovery reinit.
//!
//! ## RX audio layering
//!
//! - **A.13 receive path.** Drop the frame unless `data[1] & 0x78 == 0x78`; drop
//!   when paused; drop the silence sentinel payload `f8 ff fe`. When a session
//!   exists AND is `ready` AND the SSRC→user id is known, DAVE-decrypt the
//!   SRTP-decrypted payload with `(user_id, MediaType::Audio)`; **any** decrypt
//!   error drops the frame silently (debug-log). An unknown SSRC or missing user
//!   id passes through undecrypted to the Opus decoder. Verify songbird's receive
//!   hook applies MLS decrypt with this exact gating and error discipline.
//!
//! ## TX audio layering
//!
//! - **A.14 send path.** Layer **opus frame → DAVE `encrypt_opus` (only when the
//!   session is ready) → 12-byte RTP header (`80 78`, seq u16 BE, timestamp u32
//!   BE, ssrc u32 BE) → SRTP encrypt** via the negotiated mode. Verify songbird
//!   applies MLS encryption before RTP/SRTP framing when DAVE is active.
//!
//! These are E2EE-correctness invariants: a mis-ordered layer or a swallowed
//! error either breaks decryption for every peer or leaks plaintext. They belong
//! to the songbird-driver integration (Layer 4), not to this Layer-2 package.
