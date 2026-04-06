"""Tests for DaveVoiceWebSocket — DAVE protocol wire handling."""

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import davey
import pytest

from familiar_connect.voice.dave_ws import DaveVoiceWebSocket


@pytest.fixture
def mock_socket() -> MagicMock:
    """Mock aiohttp WebSocket."""
    sock = MagicMock()
    sock.send_str = AsyncMock()
    sock.send_bytes = AsyncMock()
    sock.close = AsyncMock()
    return sock


@pytest.fixture
def mock_connection() -> MagicMock:
    """Mock VoiceClient connection state."""
    conn = MagicMock()
    conn.server_id = 12345
    conn.user.id = 67890
    conn.session_id = "sess-abc"
    conn.token = "tok-xyz"
    # DaveVoiceClient attributes
    conn.dave_session = None
    conn.dave_protocol_version = 0
    conn._dave_pending_transitions = {}
    return conn


def _make_ws(mock_socket: MagicMock, mock_connection: MagicMock) -> DaveVoiceWebSocket:
    """Construct a DaveVoiceWebSocket with mocked dependencies."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws = DaveVoiceWebSocket(mock_socket, loop=loop)
    ws._connection = mock_connection
    ws.gateway = "wss://example.invalid/?v=8"
    return ws


# --- identify() ---


def test_identify_includes_dave_version(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """IDENTIFY payload must include max_dave_protocol_version."""
    ws = _make_ws(mock_socket, mock_connection)

    with patch.object(ws, "send_as_json", new=AsyncMock()) as send_json:
        asyncio.get_event_loop().run_until_complete(ws.identify())

    send_json.assert_called_once()
    payload = send_json.call_args.args[0]
    assert payload["op"] == ws.IDENTIFY
    assert payload["d"]["max_dave_protocol_version"] == davey.DAVE_PROTOCOL_VERSION
    # Normal fields still present
    assert payload["d"]["server_id"] == "12345"
    assert payload["d"]["user_id"] == "67890"
    assert payload["d"]["session_id"] == "sess-abc"
    assert payload["d"]["token"] == "tok-xyz"


# --- Opcode constants ---


def test_dave_opcode_constants() -> None:
    """DAVE opcodes are defined on the class."""
    assert DaveVoiceWebSocket.DAVE_PREPARE_TRANSITION == 21
    assert DaveVoiceWebSocket.DAVE_EXECUTE_TRANSITION == 22
    assert DaveVoiceWebSocket.DAVE_TRANSITION_READY == 23
    assert DaveVoiceWebSocket.DAVE_PREPARE_EPOCH == 24
    assert DaveVoiceWebSocket.MLS_EXTERNAL_SENDER == 25
    assert DaveVoiceWebSocket.MLS_KEY_PACKAGE == 26
    assert DaveVoiceWebSocket.MLS_PROPOSALS == 27
    assert DaveVoiceWebSocket.MLS_COMMIT_WELCOME == 28
    assert DaveVoiceWebSocket.MLS_ANNOUNCE_COMMIT_TRANSITION == 29
    assert DaveVoiceWebSocket.MLS_WELCOME == 30
    assert DaveVoiceWebSocket.MLS_INVALID_COMMIT_WELCOME == 31


# --- received_binary_message ---


def test_received_binary_external_sender(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 25 (EXTERNAL_SENDER) calls dave_session.set_external_sender."""
    ws = _make_ws(mock_socket, mock_connection)
    mock_connection.dave_session = MagicMock(spec=davey.DaveSession)

    payload = b"external-sender-data"
    asyncio.get_event_loop().run_until_complete(
        ws.received_binary_message(DaveVoiceWebSocket.MLS_EXTERNAL_SENDER, payload),
    )

    mock_connection.dave_session.set_external_sender.assert_called_once_with(payload)


def test_received_binary_proposals_sends_commit(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 27 (PROPOSALS) processes proposals and sends commit+welcome."""
    ws = _make_ws(mock_socket, mock_connection)
    session = MagicMock(spec=davey.DaveSession)
    commit_welcome = MagicMock()
    commit_welcome.commit = b"commit-data"
    commit_welcome.welcome = b"welcome-data"
    session.process_proposals.return_value = commit_welcome
    mock_connection.dave_session = session

    # optype byte (0=append) + proposals data
    payload = bytes([0]) + b"proposals-data"

    with patch.object(ws, "send_dave_binary", new=AsyncMock()) as send_bin:
        asyncio.get_event_loop().run_until_complete(
            ws.received_binary_message(DaveVoiceWebSocket.MLS_PROPOSALS, payload),
        )

    session.process_proposals.assert_called_once_with(
        davey.ProposalsOperationType.append,
        b"proposals-data",
    )
    send_bin.assert_called_once_with(
        DaveVoiceWebSocket.MLS_COMMIT_WELCOME,
        b"commit-data" + b"welcome-data",
    )


def test_received_binary_proposals_commit_only(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 27 with no welcome sends just the commit."""
    ws = _make_ws(mock_socket, mock_connection)
    session = MagicMock(spec=davey.DaveSession)
    commit_welcome = MagicMock()
    commit_welcome.commit = b"commit-only"
    commit_welcome.welcome = None
    session.process_proposals.return_value = commit_welcome
    mock_connection.dave_session = session

    payload = bytes([1]) + b"revoke-proposals"  # optype 1 = revoke

    with patch.object(ws, "send_dave_binary", new=AsyncMock()) as send_bin:
        asyncio.get_event_loop().run_until_complete(
            ws.received_binary_message(DaveVoiceWebSocket.MLS_PROPOSALS, payload),
        )

    session.process_proposals.assert_called_once_with(
        davey.ProposalsOperationType.revoke,
        b"revoke-proposals",
    )
    send_bin.assert_called_once_with(
        DaveVoiceWebSocket.MLS_COMMIT_WELCOME,
        b"commit-only",
    )


def test_received_binary_announce_commit(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 29 (ANNOUNCE_COMMIT) processes commit and sends TRANSITION_READY."""
    ws = _make_ws(mock_socket, mock_connection)
    session = MagicMock(spec=davey.DaveSession)
    mock_connection.dave_session = session

    transition_id = 42
    commit_data = b"commit-bytes"
    payload = struct.pack(">H", transition_id) + commit_data

    with patch.object(ws, "send_as_json", new=AsyncMock()) as send_json:
        asyncio.get_event_loop().run_until_complete(
            ws.received_binary_message(
                DaveVoiceWebSocket.MLS_ANNOUNCE_COMMIT_TRANSITION,
                payload,
            ),
        )

    session.process_commit.assert_called_once_with(commit_data)
    send_json.assert_called_once()
    sent = send_json.call_args.args[0]
    assert sent["op"] == DaveVoiceWebSocket.DAVE_TRANSITION_READY
    assert sent["d"]["transition_id"] == transition_id


def test_received_binary_welcome(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 30 (WELCOME) processes welcome and sends TRANSITION_READY."""
    ws = _make_ws(mock_socket, mock_connection)
    session = MagicMock(spec=davey.DaveSession)
    mock_connection.dave_session = session

    transition_id = 7
    welcome_data = b"welcome-bytes"
    payload = struct.pack(">H", transition_id) + welcome_data

    with patch.object(ws, "send_as_json", new=AsyncMock()) as send_json:
        asyncio.get_event_loop().run_until_complete(
            ws.received_binary_message(
                DaveVoiceWebSocket.MLS_WELCOME,
                payload,
            ),
        )

    session.process_welcome.assert_called_once_with(welcome_data)
    send_json.assert_called_once()
    sent = send_json.call_args.args[0]
    assert sent["op"] == DaveVoiceWebSocket.DAVE_TRANSITION_READY
    assert sent["d"]["transition_id"] == transition_id


def test_received_binary_commit_error_triggers_recovery(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """When process_commit raises, send INVALID_COMMIT and reinit session."""
    ws = _make_ws(mock_socket, mock_connection)
    session = MagicMock(spec=davey.DaveSession)
    session.process_commit.side_effect = ValueError("bad commit")
    mock_connection.dave_session = session
    mock_connection._reinit_dave_session = AsyncMock()

    transition_id = 9
    payload = struct.pack(">H", transition_id) + b"garbage"

    with patch.object(ws, "send_as_json", new=AsyncMock()) as send_json:
        asyncio.get_event_loop().run_until_complete(
            ws.received_binary_message(
                DaveVoiceWebSocket.MLS_ANNOUNCE_COMMIT_TRANSITION,
                payload,
            ),
        )

    send_json.assert_called_once()
    sent = send_json.call_args.args[0]
    assert sent["op"] == DaveVoiceWebSocket.MLS_INVALID_COMMIT_WELCOME
    assert sent["d"]["transition_id"] == transition_id
    mock_connection._reinit_dave_session.assert_called_once()


# --- _dispatch_binary_frame seq tracking ---


def test_dispatch_binary_frame_updates_seq_ack(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Binary frames carry a sequence number that must update seq_ack.

    py-cord's voice heartbeat sends seq_ack back to the gateway. If we
    don't update it for binary frames, the gateway considers the connection
    desynced and eventually closes it.
    """
    ws = _make_ws(mock_socket, mock_connection)
    mock_connection.dave_session = MagicMock(spec=davey.DaveSession)
    ws.seq_ack = 0

    # seq=42, op=MLS_EXTERNAL_SENDER, payload=b"data"
    raw = struct.pack(">HB", 42, DaveVoiceWebSocket.MLS_EXTERNAL_SENDER) + b"data"

    asyncio.get_event_loop().run_until_complete(ws._dispatch_binary_frame(raw))

    assert ws.seq_ack == 42


# --- send_dave_binary ---


def test_send_dave_binary_format(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """send_dave_binary frames as [op: uint8][payload].

    Outbound binary frames have NO sequence prefix — only inbound frames do.
    Sending a 2-byte zero prefix shifts the opcode by two bytes and the
    gateway interprets the resulting garbage as a protocol violation,
    closing the connection with code 4005.
    """
    ws = _make_ws(mock_socket, mock_connection)
    payload = b"key-package-bytes"

    asyncio.get_event_loop().run_until_complete(
        ws.send_dave_binary(DaveVoiceWebSocket.MLS_KEY_PACKAGE, payload),
    )

    mock_socket.send_bytes.assert_called_once()
    sent = mock_socket.send_bytes.call_args.args[0]
    assert sent[0] == DaveVoiceWebSocket.MLS_KEY_PACKAGE
    assert sent[1:] == payload


# --- JSON DAVE opcodes in received_message ---


def test_received_prepare_transition_downgrade(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 21 downgrade enables passthrough and sends TRANSITION_READY."""
    ws = _make_ws(mock_socket, mock_connection)
    session = MagicMock(spec=davey.DaveSession)
    mock_connection.dave_session = session
    mock_connection.dave_protocol_version = 1

    msg = {
        "op": DaveVoiceWebSocket.DAVE_PREPARE_TRANSITION,
        "d": {"transition_id": 5, "protocol_version": 0},
    }

    with patch.object(ws, "send_as_json", new=AsyncMock()) as send_json:
        asyncio.get_event_loop().run_until_complete(ws.received_message(msg))

    session.set_passthrough_mode.assert_called_once_with(True, 120)
    assert mock_connection._dave_pending_transitions[5] == 0
    send_json.assert_called_once()
    sent = send_json.call_args.args[0]
    assert sent["op"] == DaveVoiceWebSocket.DAVE_TRANSITION_READY
    assert sent["d"]["transition_id"] == 5


def test_received_execute_transition_applies(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 22 applies the pending transition and updates protocol version."""
    ws = _make_ws(mock_socket, mock_connection)
    mock_connection.dave_protocol_version = 0
    mock_connection._dave_pending_transitions = {3: 1}

    msg = {
        "op": DaveVoiceWebSocket.DAVE_EXECUTE_TRANSITION,
        "d": {"transition_id": 3},
    }

    asyncio.get_event_loop().run_until_complete(ws.received_message(msg))

    assert mock_connection.dave_protocol_version == 1
    assert 3 not in mock_connection._dave_pending_transitions


def test_session_description_sets_dave_protocol_version(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """SESSION_DESCRIPTION must populate dave_protocol_version on the connection.

    Without this, connect_websocket never calls _reinit_dave_session() and
    MLS binary messages arrive with dave_session=None.
    """
    ws = _make_ws(mock_socket, mock_connection)
    mock_connection.dave_protocol_version = 0

    msg = {
        "op": DaveVoiceWebSocket.SESSION_DESCRIPTION,
        "d": {
            "mode": "xsalsa20_poly1305",
            "secret_key": list(range(32)),
            "dave_protocol_version": 1,
        },
    }

    with patch.object(ws, "send_as_json", new=AsyncMock()):
        asyncio.get_event_loop().run_until_complete(ws.received_message(msg))

    assert mock_connection.dave_protocol_version == 1


def test_prepare_epoch_reinits_when_session_absent(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 24 (epoch=1) creates the DaveSession when none exists yet."""
    ws = _make_ws(mock_socket, mock_connection)
    mock_connection.dave_session = None
    mock_connection._reinit_dave_session = AsyncMock()

    msg = {
        "op": DaveVoiceWebSocket.DAVE_PREPARE_EPOCH,
        "d": {"epoch": 1, "protocol_version": 1},
    }

    asyncio.get_event_loop().run_until_complete(ws.received_message(msg))

    assert mock_connection.dave_protocol_version == 1
    mock_connection._reinit_dave_session.assert_called_once()


def test_prepare_epoch_skips_reinit_when_session_exists(
    mock_socket: MagicMock,
    mock_connection: MagicMock,
) -> None:
    """Op 24 (epoch=1) is a no-op when connect_websocket already set up the session.

    Sending a second MLS_KEY_PACKAGE desynchronises the server's MLS state and
    causes the connection to be closed (observed as close code 4005 in reconnect
    loops).
    """
    ws = _make_ws(mock_socket, mock_connection)
    existing_session = MagicMock(spec=davey.DaveSession)
    mock_connection.dave_session = existing_session
    mock_connection._reinit_dave_session = AsyncMock()

    msg = {
        "op": DaveVoiceWebSocket.DAVE_PREPARE_EPOCH,
        "d": {"epoch": 1, "protocol_version": 1},
    }

    asyncio.get_event_loop().run_until_complete(ws.received_message(msg))

    mock_connection._reinit_dave_session.assert_not_called()
    assert mock_connection.dave_session is existing_session
