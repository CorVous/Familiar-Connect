"""Deepgram streaming transcription client."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import aiohttp

from familiar_connect import log_style as ls
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from typing import Self

_logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Transcription result
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Single Deepgram streaming transcription result."""

    text: str
    is_final: bool
    start: float
    end: float
    confidence: float = 0.0
    speaker: int | None = None

    def to_message(self: Self, speaker_names: dict[int, str] | None = None) -> Message:
        """Convert to LLM Message. Prefixes content with ``[Voice]``."""
        name = "Voice"
        if self.speaker is not None and speaker_names is not None:
            name = speaker_names.get(self.speaker, "Voice")
        return Message(role="user", content=f"[Voice] {self.text}", name=name)


# TEN VAD now drives voice-activity edges locally; Deepgram is used
# for transcription text only (``Results`` messages).
TranscriptionEvent = TranscriptionResult


# ---------------------------------------------------------------------------
# Deepgram streaming transcriber
# ---------------------------------------------------------------------------


class DeepgramTranscriber:
    """Stream PCM audio to Deepgram, output :class:`TranscriptionResult` via queue."""

    def __init__(
        self: Self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = 48000,
        channels: int = 1,
        diarize: bool = False,
        interim_results: bool = False,
        utterance_end_ms: int = 1000,
        vad_events: bool = False,
        endpointing_ms: int = 300,
        replay_buffer_s: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.diarize = diarize
        self.interim_results = interim_results
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events
        self.endpointing_ms = endpointing_ms
        self.replay_buffer_s = replay_buffer_s

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        # set by ``stop()`` so the receive loop can distinguish
        # self-initiated close from server/transport closes
        self._shutting_down: bool = False
        # sliding window of recent PCM chunks; replayed to new WS on reconnect
        self._replay_buffer: collections.deque[bytes] = collections.deque()
        self._replay_buffer_bytes: int = 0
        # guards concurrent send_audio / replay-drain so bytes stay ordered
        self._send_lock: asyncio.Lock = asyncio.Lock()

    def build_ws_url(self: Self) -> str:
        """Deepgram WebSocket URL with query parameters.

        ``interim_results`` / ``utterance_end_ms`` are emitted only when
        interim results are enabled; Deepgram requires ``interim_results=true``
        for ``utterance_end_ms`` to take effect. ``endpointing`` is always
        emitted — it controls how long Deepgram waits in silence before
        finalizing a segment.
        """
        params: dict[str, str] = {
            "model": self.model,
            "language": self.language,
            "sample_rate": str(self.sample_rate),
            "channels": str(self.channels),
            "encoding": "linear16",
            "vad_events": str(self.vad_events).lower(),
            "endpointing": str(self.endpointing_ms),
        }
        if self.interim_results:
            params["interim_results"] = "true"
            params["utterance_end_ms"] = str(self.utterance_end_ms)
        if self.diarize:
            params["diarize"] = "true"
        return f"{DEEPGRAM_WS_URL}?{urlencode(params)}"

    def build_headers(self: Self) -> dict[str, str]:
        """Auth headers for Deepgram WebSocket."""
        return {"Authorization": f"Token {self.api_key}"}

    def clone(self: Self) -> DeepgramTranscriber:
        """Create transcriber with same config, independent WS connection."""
        return DeepgramTranscriber(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            sample_rate=self.sample_rate,
            channels=self.channels,
            diarize=self.diarize,
            interim_results=self.interim_results,
            utterance_end_ms=self.utterance_end_ms,
            vad_events=self.vad_events,
            endpointing_ms=self.endpointing_ms,
            replay_buffer_s=self.replay_buffer_s,
        )

    def _parse_response(self: Self, data: dict[str, Any]) -> TranscriptionResult | None:
        """Parse Deepgram response. Returns ``None`` for non-results or empty."""
        if data.get("type") != "Results":
            return None

        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return None

        best = alternatives[0]
        transcript = best.get("transcript", "")
        if not transcript:
            return None

        # extract speaker from first word if diarization is active
        speaker: int | None = None
        words = best.get("words", [])
        if words and "speaker" in words[0]:
            speaker = int(words[0]["speaker"])

        start = float(data.get("start", 0.0))
        duration = float(data.get("duration", 0.0))

        return TranscriptionResult(
            text=transcript,
            is_final=bool(data.get("is_final")),
            start=start,
            end=start + duration,
            confidence=float(best.get("confidence", 0.0)),
            speaker=speaker,
        )

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def _ws_connect(
        self: Self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
    ) -> aiohttp.ClientWebSocketResponse:
        """Open WS connection. Extracted for testability."""
        return await session.ws_connect(url, headers=headers)

    async def start(self: Self, output: asyncio.Queue[TranscriptionEvent]) -> None:
        """Connect to Deepgram and begin receiving transcription events.

        Output queue carries :class:`TranscriptionResult`s in wire order.
        VAD edges are produced elsewhere (TEN VAD) and do not flow
        through this queue.
        """
        url = self.build_ws_url()
        _logger.info(
            f"{ls.tag('🔌 WebSocket', ls.LG)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.kv('url', url, vc=ls.LW)}"
        )
        self._session = aiohttp.ClientSession()
        self._ws = await self._ws_connect(
            self._session,
            url,
            self.build_headers(),
        )
        _logger.info(
            f"{ls.tag('🔌 WebSocket', ls.G)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.kv('status', str(self._ws.close_code or 'none'), vc=ls.LW)}"
        )
        self._receive_task = asyncio.create_task(self._receive_loop(output))
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def _buffer_chunk(self: Self, data: bytes) -> None:
        """Append to sliding replay window, evicting oldest when over budget."""
        self._replay_buffer.append(data)
        self._replay_buffer_bytes += len(data)
        max_bytes = int(self.replay_buffer_s * self.sample_rate * self.channels * 2)
        while self._replay_buffer_bytes > max_bytes and self._replay_buffer:
            evicted = self._replay_buffer.popleft()
            self._replay_buffer_bytes -= len(evicted)

    async def send_audio(self: Self, data: bytes) -> None:
        """Send PCM bytes; buffer for replay on reconnect.

        Chunk is appended to the sliding replay window before attempting
        the send, so a mid-send drop still lands in the buffer.

        :raises RuntimeError: If called before :meth:`start`.
        """
        if self._ws is None:
            msg = "Transcriber is not connected — call start() first"
            raise RuntimeError(msg)
        async with self._send_lock:
            self._buffer_chunk(data)
            if self._ws.closed:
                return
            with contextlib.suppress(Exception):
                await self._ws.send_bytes(data)

    async def finalize(self: Self) -> None:
        """Force Deepgram to flush the buffered segment as a final.

        Sends ``{"type":"Finalize"}``. Discord's client-side VAD drops
        RTP during silence, so without an explicit flush Deepgram's
        endpointer never sees the silence it needs and holds the final
        until the next speech burst. Silent no-op if WS already closed
        or never started — safe to call from idle-watchdog paths.
        """
        ws = self._ws
        if ws is None or ws.closed:
            return
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "Finalize"})

    async def stop(self: Self) -> None:
        """Gracefully close Deepgram connection."""
        # flip before CloseStream so a late close frame in the receive
        # loop doesn't race into a reconnect
        self._shutting_down = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None

        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.send_json({"type": "CloseStream"})
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._ws = None

        if self._receive_task is not None:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None

        if self._session is not None:
            await self._session.close()
            self._session = None

    _MAX_RECONNECTS: int = 5
    _RECONNECT_DELAY: float = 1.0  # base delay; first attempt is immediate
    _RECONNECT_BACKOFF_CAP: float = 16.0  # max backoff in seconds
    _KEEPALIVE_INTERVAL: float = 3.0
    # extra margin on top of real-time replay duration before Finalize,
    # to absorb server-side processing jitter
    _FINALIZE_POST_REPLAY_BUFFER_S: float = 0.25

    # close codes that indicate a permanent, non-recoverable condition
    # (auth, billing, policy). reconnecting would just fail identically.
    # 1008 = policy violation (RFC 6455). 4xxx = Deepgram application-level.
    _NO_RECONNECT_CLOSE_CODES: frozenset[int] = frozenset({1008})

    @classmethod
    def _should_reconnect(cls, close_code: object) -> bool:
        """Classify close code. False → stop; True → reconnect."""
        if not isinstance(close_code, int):
            # transport-level drop (no close frame seen) — retry
            return True
        if close_code in cls._NO_RECONNECT_CLOSE_CODES:
            return False
        # Deepgram 4xxx codes = application-level errors (auth/billing/quota)
        return not (4000 <= close_code < 5000)

    async def _keepalive_loop(self: Self) -> None:
        """Periodic KeepAlive to prevent Deepgram's ~10s idle timeout.

        Re-reads ``_ws`` each tick to follow reconnects transparently.
        """
        while True:
            await asyncio.sleep(self._KEEPALIVE_INTERVAL)
            ws = self._ws
            if ws is None or ws.closed:
                continue
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "KeepAlive"})

    async def _reconnect(self: Self) -> None:
        """Close old session, open fresh one; drain replay buffer to new WS."""
        # tear down old connection
        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()

        # open new connection
        self._session = aiohttp.ClientSession()
        url = self.build_ws_url()
        self._ws = await self._ws_connect(
            self._session,
            url,
            self.build_headers(),
        )
        _logger.info(
            f"{ls.tag('🔄 WebSocket', ls.Y)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.word('reconnected', ls.W)}"
        )

        # replay buffered audio; hold send_lock so send_audio callers
        # queue behind the drain rather than interleaving
        async with self._send_lock:
            chunks_replayed = len(self._replay_buffer)
            replay_bytes = self._replay_buffer_bytes
            for chunk in self._replay_buffer:
                with contextlib.suppress(Exception):
                    await self._ws.send_bytes(chunk)
            self._replay_buffer.clear()
            self._replay_buffer_bytes = 0

        if chunks_replayed:
            # wait ~replay duration before Finalize so Deepgram can process
            # the burst. the replay arrives much faster than real-time, and
            # Finalize emits "what's been transcribed so far" — firing it
            # immediately produces a partial covering only the first few
            # chunks the server had time to process. sleep is outside the
            # send_lock so post-reconnect audio can flow through normally.
            # add a small cushion on top of real-time for server jitter.
            replay_s = replay_bytes / (self.sample_rate * self.channels * 2)
            await asyncio.sleep(replay_s + self._FINALIZE_POST_REPLAY_BUFFER_S)
            async with self._send_lock:
                ws = self._ws
                if ws is not None and not ws.closed:
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "Finalize"})
            _logger.info(
                f"{ls.tag('🔁 Replay', ls.C)} "
                f"{ls.kv('chunks', str(chunks_replayed), vc=ls.LW)}"
            )

    async def _receive_loop(
        self: Self,
        output: asyncio.Queue[TranscriptionEvent],
    ) -> None:
        """Read messages from the WebSocket, reconnecting on drops."""
        consecutive_reconnects = 0

        while consecutive_reconnects <= self._MAX_RECONNECTS:
            if self._ws is None:
                return
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    if msg_type == "Results":
                        result = self._parse_response(data)
                        if result is not None:
                            await output.put(result)
                            # got real data — reset reconnect counter
                            consecutive_reconnects = 0
                    else:
                        _logger.info(
                            f"{ls.tag('Event', ls.C)} "
                            f"{ls.word('Deepgram', ls.C)} "
                            f"{ls.kv('type', msg_type)} "
                            f"{ls.kv('data', ls.trunc(msg.data[:200], 100), vc=ls.LW)}"
                        )
                elif msg.type in {
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    _logger.warning(
                        f"{ls.tag('🔌 WebSocket', ls.Y)} "
                        f"{ls.word('Deepgram', ls.C)} "
                        f"{ls.kv('type', str(msg.type), vc=ls.LW)} "
                        f"{ls.kv('data', str(getattr(msg, 'data', None)), vc=ls.LW)}"
                    )
                    break

            close_code = self._ws.close_code if self._ws else None

            # self-initiated close → exit silently; ``stop()`` handles cleanup
            if self._shutting_down:
                _logger.info(
                    f"{ls.tag('🔌 WebSocket', ls.Y)} "
                    f"{ls.word('Deepgram', ls.C)} "
                    f"{ls.word('shutdown', ls.W)} "
                    f"{ls.kv('close_code', str(close_code), vc=ls.LW)}"
                )
                return

            # non-recoverable close (auth/billing/policy) → stop retrying
            if not self._should_reconnect(close_code):
                _logger.error(
                    f"{ls.tag('🔌 WebSocket', ls.R)} "
                    f"{ls.word('Deepgram', ls.C)} "
                    f"{ls.word('non-recoverable', ls.W)} "
                    f"{ls.kv('close_code', str(close_code), vc=ls.LW)}"
                )
                return

            outage_start = time.monotonic()
            consecutive_reconnects += 1
            # exponential backoff: first attempt is immediate; subsequent
            # failures back off as 1x, 2x, 4x... the base delay up to the cap
            backoff = 0.0
            if consecutive_reconnects > 1:
                exponent = consecutive_reconnects - 2  # attempt 2 → 2^0 = 1
                backoff = min(
                    self._RECONNECT_DELAY * (2**exponent),
                    self._RECONNECT_BACKOFF_CAP,
                )
            attempt = f"{consecutive_reconnects}/{self._MAX_RECONNECTS}"
            _logger.warning(
                f"{ls.tag('🔌 WebSocket', ls.Y)} "
                f"{ls.word('Deepgram', ls.C)} "
                f"{ls.word('reconnecting', ls.W)} "
                f"{ls.kv('close_code', str(close_code), vc=ls.LW)} "
                f"{ls.kv('attempt', attempt, vc=ls.LW)} "
                f"{ls.kv('backoff_s', f'{backoff:.1f}', vc=ls.LW)}"
            )

            try:
                if backoff > 0:
                    await asyncio.sleep(backoff)
                await self._reconnect()
                outage_s = time.monotonic() - outage_start
                _logger.info(
                    f"{ls.tag('🔌 WebSocket', ls.G)} "
                    f"{ls.word('Deepgram', ls.C)} "
                    f"{ls.word('recovered', ls.LG)} "
                    f"{ls.kv('close_code', str(close_code), vc=ls.LW)} "
                    f"{ls.kv('attempt', attempt, vc=ls.LW)} "
                    f"{ls.kv('outage_s', f'{outage_s:.2f}', vc=ls.LW)}"
                )
            except Exception:
                _logger.exception("Deepgram reconnection failed")
                return

        _logger.error(
            f"{ls.tag('🔌 WebSocket', ls.R)} "
            f"{ls.word('Deepgram', ls.C)} "
            f"{ls.word('max reconnects exhausted', ls.W)} "
            f"{ls.kv('attempts', str(self._MAX_RECONNECTS), vc=ls.LW)}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _env_float(raw: str | None, default: float) -> float:
    """Parse *raw* as float, falling back to *default*."""
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(raw: str | None, default: int) -> int:
    """Parse *raw* as int, falling back to *default*."""
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def create_transcriber_from_env() -> DeepgramTranscriber:
    """Create from env vars (``DEEPGRAM_API_KEY`` required).

    Optional overrides:

    - ``DEEPGRAM_MODEL`` (default ``nova-3``)
    - ``DEEPGRAM_LANGUAGE`` (default ``en``)
    - ``DEEPGRAM_REPLAY_BUFFER_S`` (default ``5.0``)
    - ``DEEPGRAM_KEEPALIVE_INTERVAL_S`` (default ``3.0``)
    - ``DEEPGRAM_RECONNECT_MAX_ATTEMPTS`` (default ``5``)
    - ``DEEPGRAM_RECONNECT_BACKOFF_CAP_S`` (default ``16.0``)

    :raises ValueError: If ``DEEPGRAM_API_KEY`` not set.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        msg = "DEEPGRAM_API_KEY environment variable is required"
        raise ValueError(msg)

    model = os.environ.get("DEEPGRAM_MODEL") or DEFAULT_MODEL
    language = os.environ.get("DEEPGRAM_LANGUAGE") or DEFAULT_LANGUAGE
    replay_buffer_s = _env_float(os.environ.get("DEEPGRAM_REPLAY_BUFFER_S"), 5.0)
    keepalive_interval_s = _env_float(
        os.environ.get("DEEPGRAM_KEEPALIVE_INTERVAL_S"), 3.0
    )
    max_attempts = _env_int(os.environ.get("DEEPGRAM_RECONNECT_MAX_ATTEMPTS"), 5)
    backoff_cap_s = _env_float(os.environ.get("DEEPGRAM_RECONNECT_BACKOFF_CAP_S"), 16.0)

    t = DeepgramTranscriber(
        api_key=api_key,
        model=model,
        language=language,
        replay_buffer_s=replay_buffer_s,
    )
    t._KEEPALIVE_INTERVAL = keepalive_interval_s  # noqa: SLF001
    t._MAX_RECONNECTS = max_attempts  # noqa: SLF001
    t._RECONNECT_BACKOFF_CAP = backoff_cap_s  # noqa: SLF001
    return t
