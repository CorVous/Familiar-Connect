"""DAVE-aware voice WebSocket for py-cord.

Subclasses py-cord's ``DiscordVoiceWebSocket`` to add:

* ``max_dave_protocol_version`` in IDENTIFY payloads
* Binary WebSocket frame handling (py-cord silently drops them)
* Dispatch for DAVE voice gateway opcodes 21-31
* MLS handshake driven by ``davey.DaveSession`` state on the voice client

See the DAVE protocol whitepaper at https://daveprotocol.com/ and the
davey library usage guide at https://github.com/Snazzah/davey/blob/master/docs/USAGE.md.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import TYPE_CHECKING, Any

import aiohttp
import davey
from discord import utils
from discord.errors import ConnectionClosed
from discord.gateway import DiscordVoiceWebSocket

if TYPE_CHECKING:
    from collections.abc import Mapping

_logger = logging.getLogger(__name__)

# Binary frame header: [seq: uint16BE][op: uint8][payload...]
_BINARY_HEADER_STRUCT = struct.Struct(">HB")
# Transition ID prefix on ops 29 and 30: [transition_id: uint16BE]
_TRANSITION_ID_STRUCT = struct.Struct(">H")


class DaveVoiceWebSocket(DiscordVoiceWebSocket):
    """Voice WebSocket with DAVE protocol support."""

    # Declared for type-checkers; the base class sets these at runtime.
    _connection: Any
    gateway: str

    # DAVE opcodes (JSON)
    DAVE_PREPARE_TRANSITION = 21
    DAVE_EXECUTE_TRANSITION = 22
    DAVE_TRANSITION_READY = 23
    DAVE_PREPARE_EPOCH = 24

    # DAVE/MLS opcodes (Binary)
    MLS_EXTERNAL_SENDER = 25
    MLS_KEY_PACKAGE = 26
    MLS_PROPOSALS = 27
    MLS_COMMIT_WELCOME = 28
    MLS_ANNOUNCE_COMMIT_TRANSITION = 29
    MLS_WELCOME = 30
    MLS_INVALID_COMMIT_WELCOME = 31

    # --- IDENTIFY with DAVE version ---

    async def identify(self) -> None:
        """Send IDENTIFY advertising DAVE protocol support.

        Without ``max_dave_protocol_version``, Discord's voice gateway
        assumes the client has no E2EE support and closes the connection
        with code 4017 as of March 2026.
        """
        state = self._connection
        payload = {
            "op": self.IDENTIFY,
            "d": {
                "server_id": str(state.server_id),
                "user_id": str(state.user.id),
                "session_id": state.session_id,
                "token": state.token,
                "max_dave_protocol_version": davey.DAVE_PROTOCOL_VERSION,
            },
        }
        await self.send_as_json(payload)

    # --- Polling with binary support ---

    async def poll_event(self) -> None:
        """Receive next WebSocket frame, dispatching binary and text.

        py-cord's implementation only handles TEXT frames, so DAVE binary
        opcodes would otherwise be silently dropped.
        """
        msg = await asyncio.wait_for(self.ws.receive(), timeout=30.0)
        if msg.type is aiohttp.WSMsgType.TEXT:
            await self.received_message(utils._from_json(msg.data))  # noqa: SLF001 — discord.utils._from_json is the standard decode helper; no public alternative is provided by py-cord
        elif msg.type is aiohttp.WSMsgType.BINARY:
            await self._dispatch_binary_frame(msg.data)
        elif msg.type is aiohttp.WSMsgType.ERROR:
            _logger.debug("Voice WS ERROR frame: %s", msg)
            raise ConnectionClosed(self.ws, shard_id=None) from msg.data
        elif msg.type in {
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        }:
            _logger.debug("Voice WS closed: %s", msg)
            raise ConnectionClosed(self.ws, shard_id=None, code=self._close_code)

    async def _dispatch_binary_frame(self, raw: bytes) -> None:
        """Parse a DAVE binary frame and dispatch to received_binary_message.

        Binary frames carry a sequence number in the first two bytes that
        must update ``seq_ack`` so py-cord's heartbeat can acknowledge it.
        Without this, the gateway treats the connection as desynced and
        eventually closes it (which leads to a reconnect storm).
        """
        if len(raw) < _BINARY_HEADER_STRUCT.size:
            _logger.warning("Voice binary frame too short: %d bytes", len(raw))
            return
        seq, op = _BINARY_HEADER_STRUCT.unpack_from(raw, 0)
        self.seq_ack = seq
        payload = raw[_BINARY_HEADER_STRUCT.size :]
        await self.received_binary_message(op, payload)

    # --- JSON message handling (DAVE extensions) ---

    async def received_message(self, msg: Mapping[str, Any]) -> None:
        """Handle JSON voice gateway messages, including DAVE opcodes.

        super().received_message() is always called so that py-cord can update
        its ``seq_ack`` counter (used for buffered resume) on every message,
        including DAVE-specific opcodes that the base class doesn't know about.
        """
        op = msg.get("op")
        if op == self.SESSION_DESCRIPTION:
            # Read dave_protocol_version before super() processes the message,
            # so connect_websocket can call _reinit_dave_session() immediately after
            # the secret_key is set — i.e. before binary MLS messages arrive.
            data = msg.get("d") or {}
            version = int(data.get("dave_protocol_version", 0))
            self._connection.dave_protocol_version = version
            _logger.debug("SESSION_DESCRIPTION dave_protocol_version=%d", version)
        elif op == self.DAVE_PREPARE_TRANSITION:
            await self._handle_prepare_transition(msg.get("d", {}))
        elif op == self.DAVE_EXECUTE_TRANSITION:
            await self._handle_execute_transition(msg.get("d", {}))
        elif op == self.DAVE_PREPARE_EPOCH:
            await self._handle_prepare_epoch(msg.get("d", {}))

        await super().received_message(msg)

    async def _handle_prepare_transition(self, data: Mapping[str, Any]) -> None:
        """Op 21: store pending transition; enable passthrough if downgrading."""
        transition_id = int(data["transition_id"])
        protocol_version = int(data["protocol_version"])
        conn = self._connection
        conn._dave_pending_transitions[transition_id] = protocol_version  # noqa: SLF001 — DaveVoiceWebSocket is the wire layer for DaveVoiceClient; it owns _dave_pending_transitions by design

        if protocol_version == 0 and conn.dave_session is not None:
            conn.dave_session.set_passthrough_mode(True, 120)  # noqa: FBT003 — davey API; bool arg is positional and cannot be refactored away

        await self.send_as_json({
            "op": self.DAVE_TRANSITION_READY,
            "d": {"transition_id": transition_id},
        })

    async def _handle_execute_transition(self, data: Mapping[str, Any]) -> None:
        """Op 22: apply the pending transition."""
        transition_id = int(data["transition_id"])
        conn = self._connection
        pending = conn._dave_pending_transitions.pop(transition_id, None)  # noqa: SLF001 — see _handle_prepare_transition; DaveVoiceWebSocket manages this dict on behalf of DaveVoiceClient
        if pending is None:
            _logger.debug(
                "DAVE EXECUTE_TRANSITION for unknown transition_id %d"
                " (likely initial handshake)",
                transition_id,
            )
            return
        conn.dave_protocol_version = pending
        _logger.info(
            "DAVE transition %d executed — protocol version now %d",
            transition_id,
            pending,
        )

    async def _handle_prepare_epoch(self, data: Mapping[str, Any]) -> None:
        """Op 24: reinit DaveSession for new MLS group at epoch 1.

        Only initializes when no session exists yet. If SESSION_DESCRIPTION
        already triggered initialization via connect_websocket, this is a
        no-op — sending a second MLS_KEY_PACKAGE would desync the MLS state
        and cause the server to close the connection.
        """
        epoch = int(data.get("epoch", 0))
        protocol_version = int(data.get("protocol_version", 0))
        conn = self._connection
        if epoch == 1 and conn.dave_session is None:
            conn.dave_protocol_version = protocol_version
            await conn._reinit_dave_session()  # noqa: SLF001 — the WS layer drives session lifecycle; _reinit_dave_session is a semi-internal hook shared between these two companion classes

    # --- Binary message handling ---

    async def received_binary_message(self, op: int, data: bytes) -> None:
        """Dispatch a DAVE binary opcode to its handler."""
        if op == self.MLS_EXTERNAL_SENDER:
            await self._handle_external_sender(data)
        elif op == self.MLS_PROPOSALS:
            await self._handle_proposals(data)
        elif op == self.MLS_ANNOUNCE_COMMIT_TRANSITION:
            await self._handle_announce_commit(data)
        elif op == self.MLS_WELCOME:
            await self._handle_welcome(data)
        else:
            _logger.debug("Unhandled DAVE binary opcode %d (%d bytes)", op, len(data))

    async def _handle_external_sender(self, data: bytes) -> None:
        """Op 25: install the voice gateway's MLS external sender credential."""
        session = self._connection.dave_session
        if session is None:
            _logger.warning("MLS_EXTERNAL_SENDER received with no dave_session")
            return
        session.set_external_sender(data)
        _logger.debug("DAVE external sender installed (%d bytes)", len(data))

    async def _handle_proposals(self, data: bytes) -> None:
        """Op 27: process add/remove proposals and emit commit+welcome if any."""
        if len(data) < 1:
            _logger.warning("MLS_PROPOSALS payload too short")
            return
        session = self._connection.dave_session
        if session is None:
            _logger.warning("MLS_PROPOSALS received with no dave_session")
            return

        optype_byte = data[0]
        proposals_data = data[1:]
        optype = (
            davey.ProposalsOperationType.append
            if optype_byte == 0
            else davey.ProposalsOperationType.revoke
        )

        result = session.process_proposals(optype, proposals_data)
        if result is None:
            return

        commit_bytes = result.commit
        welcome_bytes = result.welcome
        frame = (
            commit_bytes + welcome_bytes if welcome_bytes is not None else commit_bytes
        )
        await self.send_dave_binary(self.MLS_COMMIT_WELCOME, frame)

    async def _handle_announce_commit(self, data: bytes) -> None:
        """Op 29: apply a server-announced commit and report ready."""
        if len(data) < _TRANSITION_ID_STRUCT.size:
            _logger.warning("MLS_ANNOUNCE_COMMIT payload too short")
            return
        (transition_id,) = _TRANSITION_ID_STRUCT.unpack_from(data, 0)
        commit_data = data[_TRANSITION_ID_STRUCT.size :]

        session = self._connection.dave_session
        if session is None:
            _logger.warning("MLS_ANNOUNCE_COMMIT received with no dave_session")
            return

        try:
            session.process_commit(commit_data)
        except (ValueError, RuntimeError) as exc:
            _logger.warning("process_commit failed, recovering: %s", exc)
            await self._report_invalid_and_recover(transition_id)
            return

        await self.send_as_json({
            "op": self.DAVE_TRANSITION_READY,
            "d": {"transition_id": transition_id},
        })

    async def _handle_welcome(self, data: bytes) -> None:
        """Op 30: apply an MLS welcome and report ready."""
        if len(data) < _TRANSITION_ID_STRUCT.size:
            _logger.warning("MLS_WELCOME payload too short")
            return
        (transition_id,) = _TRANSITION_ID_STRUCT.unpack_from(data, 0)
        welcome_data = data[_TRANSITION_ID_STRUCT.size :]

        session = self._connection.dave_session
        if session is None:
            _logger.warning("MLS_WELCOME received with no dave_session")
            return

        try:
            session.process_welcome(welcome_data)
        except (ValueError, RuntimeError) as exc:
            _logger.warning("process_welcome failed, recovering: %s", exc)
            await self._report_invalid_and_recover(transition_id)
            return

        await self.send_as_json({
            "op": self.DAVE_TRANSITION_READY,
            "d": {"transition_id": transition_id},
        })

    async def _report_invalid_and_recover(self, transition_id: int) -> None:
        """Tell the gateway we can't process a commit/welcome and reinit."""
        await self.send_as_json({
            "op": self.MLS_INVALID_COMMIT_WELCOME,
            "d": {"transition_id": transition_id},
        })
        await self._connection._reinit_dave_session()  # noqa: SLF001 — recovery path; DaveVoiceWebSocket triggers session reinitialization on the owning DaveVoiceClient by design

    # --- Binary send helper ---

    async def send_dave_binary(self, op: int, payload: bytes) -> None:
        """Frame and send a DAVE binary opcode.

        Format: ``[op: uint8][payload...]``.

        Note: only *received* binary frames carry the 2-byte sequence
        prefix; outbound frames omit it (matching the discord.py reference
        implementation in PR #10300). Sending a 2-byte zero prefix shifts
        the opcode and corrupts the payload, which the gateway responds to
        by closing the connection (observed as close code 4005).
        """
        frame = bytes([op]) + payload
        await self.ws.send_bytes(frame)
