"""Tests for DaveVoiceClient — DaveSession lifecycle management."""

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import davey
import pytest

from familiar_connect.voice.dave_client import DaveVoiceClient
from familiar_connect.voice.dave_ws import DaveVoiceWebSocket


class _TestClient(DaveVoiceClient):
    """Subclass that bypasses VoiceClient.__init__ and stubs user/channel."""

    def __init__(self) -> None:
        self.dave_session = None
        self.dave_protocol_version = 0
        self._dave_pending_transitions = {}
        self._user = MagicMock(id=67890)
        self._channel = MagicMock(id=12345)
        self.ws = MagicMock()
        self.ws.send_dave_binary = AsyncMock()
        self._connected = MagicMock()
        # Fields needed by _get_voice_packet
        self.sequence = 1
        self.timestamp = 100
        self.ssrc = 42
        self.mode = "aead_xchacha20_poly1305_rtpsize"

    @property
    def user(self):  # type: ignore[override]
        return self._user

    @property
    def channel(self):  # type: ignore[override]
        return self._channel


@pytest.fixture
def client() -> DaveVoiceClient:
    """Construct a DaveVoiceClient without running VoiceClient.__init__."""
    return _TestClient()


def test_connect_websocket_uses_dave_ws(client: DaveVoiceClient) -> None:
    """connect_websocket builds a DaveVoiceWebSocket via from_client."""
    fake_ws = MagicMock()
    fake_ws.secret_key = b"key"
    fake_ws.poll_event = AsyncMock()

    with patch.object(
        DaveVoiceWebSocket,
        "from_client",
        new=AsyncMock(return_value=fake_ws),
    ) as from_client:
        result = asyncio.new_event_loop().run_until_complete(
            client.connect_websocket(),
        )

    from_client.assert_called_once_with(client)
    assert result is fake_ws
    client._connected.clear.assert_called_once()  # ty: ignore[unresolved-attribute]
    client._connected.set.assert_called_once()  # ty: ignore[unresolved-attribute]


def test_connect_websocket_reinits_when_dave_negotiated(
    client: DaveVoiceClient,
) -> None:
    """connect_websocket passes the local ws to _reinit_dave_session.

    self.ws is not set until connect_websocket returns (py-cord assigns it
    to the return value), so the local ws object must be forwarded explicitly.
    """
    client.dave_protocol_version = 1
    fake_ws = MagicMock()
    fake_ws.secret_key = b"key"
    fake_ws.poll_event = AsyncMock()

    with (
        patch.object(
            DaveVoiceWebSocket,
            "from_client",
            new=AsyncMock(return_value=fake_ws),
        ),
        patch.object(
            client,
            "_reinit_dave_session",
            new=AsyncMock(),
        ) as reinit,
    ):
        asyncio.new_event_loop().run_until_complete(client.connect_websocket())

    reinit.assert_called_once_with(fake_ws)


def test_reinit_dave_session_uses_supplied_ws(
    client: DaveVoiceClient,
) -> None:
    """When ws is supplied, _reinit_dave_session sends on that ws, not self.ws.

    This exercises the initial-connect path where self.ws is still the sentinel.
    """
    client.dave_protocol_version = 1
    fake_session = MagicMock(spec=davey.DaveSession)
    fake_session.get_serialized_key_package.return_value = b"kp-bytes"
    supplied_ws = MagicMock()
    supplied_ws.send_dave_binary = AsyncMock()

    with patch.object(davey, "DaveSession", return_value=fake_session) as ctor:
        asyncio.new_event_loop().run_until_complete(
            client._reinit_dave_session(supplied_ws),
        )

    ctor.assert_called_once_with(1, 67890, 12345)
    assert client.dave_session is fake_session
    supplied_ws.send_dave_binary.assert_called_once_with(
        DaveVoiceWebSocket.MLS_KEY_PACKAGE,
        b"kp-bytes",
    )
    # self.ws must NOT be touched — it's the sentinel during initial connect
    client.ws.send_dave_binary.assert_not_called()  # ty: ignore[unresolved-attribute]


def test_reinit_dave_session_falls_back_to_self_ws(
    client: DaveVoiceClient,
) -> None:
    """When ws is omitted, _reinit_dave_session falls back to self.ws.

    This exercises the recovery path where the connection is already established.
    """
    client.dave_protocol_version = 1
    fake_session = MagicMock(spec=davey.DaveSession)
    fake_session.get_serialized_key_package.return_value = b"kp-bytes"

    with patch.object(davey, "DaveSession", return_value=fake_session) as ctor:
        asyncio.new_event_loop().run_until_complete(client._reinit_dave_session())

    ctor.assert_called_once_with(1, 67890, 12345)
    assert client.dave_session is fake_session
    client.ws.send_dave_binary.assert_called_once_with(  # ty: ignore[unresolved-attribute]
        DaveVoiceWebSocket.MLS_KEY_PACKAGE,
        b"kp-bytes",
    )


# ---------------------------------------------------------------------------
# _get_voice_packet — DAVE encryption layer
# ---------------------------------------------------------------------------


def _srtp_passthrough(
    _self: object,
    header: bytes,
    data: bytes,
) -> bytes:
    """Stub SRTP encryption that concatenates header + data."""
    return bytes(header) + bytes(data)


_SRTP_ATTR = "_encrypt_aead_xchacha20_poly1305_rtpsize"


class TestGetVoicePacket:
    """Tests for _get_voice_packet DAVE encryption integration."""

    def _make_client(self) -> _TestClient:
        return _TestClient()

    def test_no_dave_session_passes_data_through(self) -> None:
        """Without a DAVE session, opus data is passed to SRTP unmodified."""
        c = self._make_client()
        c.dave_session = None
        opus_frame = b"\xaa\xbb\xcc\xdd"

        with patch.object(
            type(c),
            _SRTP_ATTR,
            _srtp_passthrough,
        ):
            packet = c._get_voice_packet(opus_frame)

        # First 12 bytes are the RTP header, remainder is the opus data.
        assert packet[12:] == opus_frame

    def test_dave_not_ready_passes_data_through(self) -> None:
        """When DAVE session exists but is not ready, data passes through."""
        c = self._make_client()
        c.dave_session = MagicMock()
        c.dave_session.ready = False
        opus_frame = b"\xaa\xbb\xcc\xdd"

        with patch.object(
            type(c),
            _SRTP_ATTR,
            _srtp_passthrough,
        ):
            packet = c._get_voice_packet(opus_frame)

        assert packet[12:] == opus_frame
        c.dave_session.encrypt_opus.assert_not_called()

    def test_dave_ready_encrypts_opus_frame(self) -> None:
        """When DAVE is ready, encrypt_opus is called on the opus frame."""
        c = self._make_client()
        c.dave_session = MagicMock()
        c.dave_session.ready = True
        c.dave_session.encrypt_opus.return_value = b"\xee\xff"
        opus_frame = b"\xaa\xbb\xcc\xdd"

        with patch.object(
            type(c),
            _SRTP_ATTR,
            _srtp_passthrough,
        ):
            packet = c._get_voice_packet(opus_frame)

        c.dave_session.encrypt_opus.assert_called_once_with(opus_frame)
        # The encrypted payload should appear after the RTP header.
        assert packet[12:] == b"\xee\xff"

    def test_rtp_header_fields(self) -> None:
        """RTP header has correct version, type, seq, ts, ssrc."""
        c = self._make_client()
        c.sequence = 0x1234
        c.timestamp = 0xDEADBEEF
        c.ssrc = 0x00C0FFEE

        with patch.object(
            type(c),
            _SRTP_ATTR,
            _srtp_passthrough,
        ):
            packet = c._get_voice_packet(b"\x00")
        header = packet[:12]

        assert header[0] == 0x80  # version 2
        assert header[1] == 0x78  # payload type
        assert struct.unpack_from(">H", header, 2)[0] == 0x1234
        assert struct.unpack_from(">I", header, 4)[0] == 0xDEADBEEF
        assert struct.unpack_from(">I", header, 8)[0] == 0x00C0FFEE

    def test_encrypt_opus_receives_bytes(self) -> None:
        """encrypt_opus receives bytes, not bytearray (davey requirement)."""
        c = self._make_client()
        c.dave_session = MagicMock()
        c.dave_session.ready = True
        c.dave_session.encrypt_opus.return_value = b"\x00"

        with patch.object(
            type(c),
            _SRTP_ATTR,
            _srtp_passthrough,
        ):
            c._get_voice_packet(b"\xaa\xbb")

        args = c.dave_session.encrypt_opus.call_args[0]
        assert isinstance(args[0], bytes)
