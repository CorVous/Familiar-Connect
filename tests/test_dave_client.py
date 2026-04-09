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


# ---------------------------------------------------------------------------
# unpack_audio — DAVE decryption layer
# ---------------------------------------------------------------------------


class TestUnpackAudioDaveDecrypt:
    """Tests for DAVE decryption in the receive path."""

    # Minimal valid RTP header: version=2 (0x80), payload type=0x78
    _RTP_HEADER = b"\x80\x78" + b"\x00" * 12

    def _make_raw_data(
        self,
        *,
        ssrc: int = 42,
        decrypted_data: bytes = b"\xaa\xbb\xcc\xdd",
    ) -> MagicMock:
        """Create a mock RawData with controllable decrypted_data."""
        raw = MagicMock()
        raw.ssrc = ssrc
        raw.decrypted_data = decrypted_data
        return raw

    def test_dave_decrypts_audio_on_receive(self) -> None:
        """When DAVE is ready, unpack_audio DAVE-decrypts the payload."""
        c = _TestClient()
        c.paused = False
        c.dave_session = MagicMock()
        c.dave_session.ready = True
        c.dave_session.decrypt.return_value = b"\x01\x02"
        c.decoder = MagicMock()
        c.ws = MagicMock()
        c.ws.ssrc_map = {42: {"user_id": 12345}}

        raw = self._make_raw_data(ssrc=42, decrypted_data=b"\xaa\xbb")

        with patch("familiar_connect.voice.dave_client.RawData", return_value=raw):
            c.unpack_audio(self._RTP_HEADER)

        c.dave_session.decrypt.assert_called_once_with(
            12345, davey.MediaType.audio, b"\xaa\xbb"
        )
        assert raw.decrypted_data == b"\x01\x02"

    def test_no_dave_session_skips_decrypt(self) -> None:
        """Without a DAVE session, decrypted_data passes through unchanged."""
        c = _TestClient()
        c.paused = False
        c.dave_session = None
        c.decoder = MagicMock()
        c.ws = MagicMock()

        raw = self._make_raw_data(decrypted_data=b"\xaa\xbb")

        with patch("familiar_connect.voice.dave_client.RawData", return_value=raw):
            c.unpack_audio(self._RTP_HEADER)

        assert raw.decrypted_data == b"\xaa\xbb"

    def test_dave_not_ready_skips_decrypt(self) -> None:
        """When DAVE session exists but is not ready, data passes through."""
        c = _TestClient()
        c.paused = False
        c.dave_session = MagicMock()
        c.dave_session.ready = False
        c.decoder = MagicMock()
        c.ws = MagicMock()

        raw = self._make_raw_data(decrypted_data=b"\xaa\xbb")

        with patch("familiar_connect.voice.dave_client.RawData", return_value=raw):
            c.unpack_audio(self._RTP_HEADER)

        c.dave_session.decrypt.assert_not_called()
        assert raw.decrypted_data == b"\xaa\xbb"

    def test_unknown_ssrc_skips_decrypt(self) -> None:
        """When the SSRC is not in the ssrc_map, DAVE decrypt is skipped."""
        c = _TestClient()
        c.paused = False
        c.dave_session = MagicMock()
        c.dave_session.ready = True
        c.decoder = MagicMock()
        c.ws = MagicMock()
        c.ws.ssrc_map = {}

        raw = self._make_raw_data(ssrc=999, decrypted_data=b"\xaa\xbb")

        with patch("familiar_connect.voice.dave_client.RawData", return_value=raw):
            c.unpack_audio(self._RTP_HEADER)

        c.dave_session.decrypt.assert_not_called()

    def test_dave_decrypt_error_skips_frame(self) -> None:
        """When DAVE decrypt fails, the frame is dropped (not opus-decoded)."""
        c = _TestClient()
        c.paused = False
        c.dave_session = MagicMock()
        c.dave_session.ready = True
        c.dave_session.decrypt.side_effect = RuntimeError("bad frame")
        c.decoder = MagicMock()
        c.ws = MagicMock()
        c.ws.ssrc_map = {42: {"user_id": 12345}}

        raw = self._make_raw_data(ssrc=42, decrypted_data=b"\xaa\xbb")

        with patch("familiar_connect.voice.dave_client.RawData", return_value=raw):
            c.unpack_audio(self._RTP_HEADER)

        c.decoder.decode.assert_not_called()
