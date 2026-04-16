"""DAVE-enabled voice client for py-cord.

Subclasses ``discord.VoiceClient`` to manage a ``davey.DaveSession``
lifecycle alongside the standard voice connection. The DAVE wire protocol
is handled by :class:`DaveVoiceWebSocket`; this class owns the session
state and drives session (re)initialization.
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

import davey
from discord import VoiceClient
from discord.sinks.core import RawData

from familiar_connect import log_style as ls
from familiar_connect.voice.dave_ws import DaveVoiceWebSocket

if TYPE_CHECKING:
    from discord import Client
    from discord.abc import Connectable

_logger = logging.getLogger(__name__)


class DaveVoiceClient(VoiceClient):
    """Voice client that negotiates DAVE E2EE with the voice gateway."""

    def __init__(self, client: Client, channel: Connectable) -> None:
        """Initialize DAVE state alongside the base voice client."""
        super().__init__(client, channel)
        self.dave_session: davey.DaveSession | None = None
        self.dave_protocol_version: int = 0
        self._dave_pending_transitions: dict[int, int] = {}

    async def connect_websocket(self) -> DaveVoiceWebSocket:
        """Create the DAVE-aware voice WebSocket and run the handshake.

        Mirrors py-cord's ``VoiceClient.connect_websocket``: construct the
        WebSocket, then poll until ``secret_key`` is available. After that,
        initialize the DaveSession if the server negotiated a non-zero
        protocol version.
        """
        ws = await DaveVoiceWebSocket.from_client(self)
        self._connected.clear()
        while ws.secret_key is None:
            await ws.poll_event()
        self._connected.set()

        if self.dave_protocol_version > 0:
            await self._reinit_dave_session(ws)

        return ws

    async def _reinit_dave_session(self, ws: DaveVoiceWebSocket | None = None) -> None:
        """Create a fresh DaveSession and send our MLS key package.

        Called on initial connection (after SESSION_DESCRIPTION arrives with a
        non-zero dave_protocol_version) and on recovery from an invalid
        commit/welcome.

        :param ws: The WebSocket to send on. When called from
            ``connect_websocket`` the ``self.ws`` attribute is not yet
            set (py-cord assigns it after this method returns), so the
            caller must supply the WebSocket explicitly. Recovery calls
            (from :class:`DaveVoiceWebSocket`) leave this as ``None`` and
            fall back to ``self.ws``, which is set by then.
        """
        target_ws = ws if ws is not None else self.ws
        user_id = int(self.user.id)
        channel_id = int(self.channel.id)  # ty: ignore[unresolved-attribute]
        self.dave_session = davey.DaveSession(
            self.dave_protocol_version,
            user_id,
            channel_id,
        )
        _logger.info(
            f"{ls.tag('🔐 DAVE', ls.LB)} "
            f"{ls.kv('event', 'initialized', vc=ls.LW)} "
            f"{ls.kv('protocol_version', str(self.dave_protocol_version), vc=ls.LW)} "
            f"{ls.kv('user', str(user_id), vc=ls.LC)}"
        )

        key_package = self.dave_session.get_serialized_key_package()
        await target_ws.send_dave_binary(  # ty: ignore[unresolved-attribute] — target_ws is DaveVoiceWebSocket at runtime; ty sees the VoiceClient.ws type as DiscordVoiceWebSocket
            DaveVoiceWebSocket.MLS_KEY_PACKAGE,
            key_package,
        )

    def unpack_audio(self, data: bytes) -> None:
        """Unpack received audio, inserting DAVE decryption before opus decode.

        The base class decrypts SRTP and constructs a :class:`RawData` whose
        ``decrypted_data`` is the inner payload. With DAVE, that payload is
        still DAVE-encrypted. This override decrypts it before the opus decoder
        sees it.
        """
        if data[1] & 0x78 != 0x78:
            return
        if self.paused:
            return

        raw = RawData(data, self)
        if raw.decrypted_data == b"\xf8\xff\xfe":
            return

        if (
            self.dave_session is not None
            and self.dave_session.ready
            and raw.decrypted_data is not None
        ):
            ssrc_info = self.ws.ssrc_map.get(raw.ssrc)
            if ssrc_info is not None:
                user_id = ssrc_info.get("user_id")
                if user_id is not None:
                    try:
                        raw.decrypted_data = self.dave_session.decrypt(
                            user_id, davey.MediaType.audio, raw.decrypted_data
                        )
                    except Exception:  # noqa: BLE001
                        _logger.debug(
                            "DAVE decrypt failed for SSRC %d, dropping frame",
                            raw.ssrc,
                        )
                        return

        if self.decoder is not None:
            self.decoder.decode(raw)

    def _get_voice_packet(self, data: bytes) -> bytes:
        """Build an RTP packet, applying DAVE encryption before SRTP.

        DAVE encrypts the raw opus frame *before* it is placed into the
        RTP packet and SRTP-encrypted. The layering is:
        opus frame → DAVE encrypt → RTP + SRTP.
        """
        payload = data
        if self.dave_session is not None and self.dave_session.ready:
            payload = self.dave_session.encrypt_opus(bytes(data))

        header = bytearray(12)
        header[0] = 0x80
        header[1] = 0x78
        struct.pack_into(">H", header, 2, self.sequence)
        struct.pack_into(">I", header, 4, self.timestamp)
        struct.pack_into(">I", header, 8, self.ssrc)

        encrypt_packet = getattr(self, f"_encrypt_{self.mode}")
        return encrypt_packet(header, payload)
